import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import yaml
from dataclasses import dataclass, field
import logging
import time
import json

# --- Настройка импортов ---
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '')))

from src.spore_tree_config import SporeTreeConfig
from src.spore_tree import SporeTree
from src.pendulum import PendulumSystem
from src.tree_evaluator import TreeEvaluator
from src.visualize_spore_tree import visualize_spore_tree
from src.matching.soft_assignment import pairwise_sqdist

@dataclass
class OptimizationState:
    iteration: int = 0
    history: list = field(default_factory=list)

def setup_logging(log_dir: str):
    log_file = os.path.join(log_dir, 'optimization.log')
    
    logger = logging.getLogger()
    # Закрываем и удаляем существующие обработчики
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout) 
        ]
    )
    logging.info(f"Логгер инициализирован. Логи сохраняются в {log_file}")

def main():
    # --- Создание директории для результатов ---
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join('runs', run_timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # --- Настройка логирования ---
    setup_logging(run_dir)

    # --- Загрузка конфигурации ---
    try:
        with open('config/optimization.yaml', 'r', encoding='utf-8') as f:
            config_yaml = yaml.safe_load(f)
        logging.info("✅ Конфигурация успешно загружена из 'config/optimization.yaml'.")
    except FileNotFoundError:
        logging.error("Ошибка: Файл 'config/optimization.yaml' не найден.")
        return

    opt_config = config_yaml['optimizer']
    integ_config = config_yaml['integration']
    
    # --- Инициализация системы ---
    pendulum = PendulumSystem()
    spore_tree_config = SporeTreeConfig(
        initial_position=np.array([np.pi, 0.0]),
        dt_base=integ_config['step_size'],
        dt_grandchildren_factor=0.1,
        show_debug=False
    )
    tree = SporeTree(pendulum, spore_tree_config)
    evaluator = TreeEvaluator(tree)

    num_children = 4
    num_grandchildren_per_child = 2
    num_total_grandchildren = num_children * num_grandchildren_per_child
    num_params = num_total_grandchildren
    logging.info(f"Система инициализирована. Количество оптимизируемых параметров (только внуки): {num_params}")

    opt_state = OptimizationState()

    def grandchildren_pairing_loss(dt_grandchildren, fixed_dt_children, pairing_map):
        dt_all = np.concatenate([fixed_dt_children, dt_grandchildren])
        evaluator._build_if_needed(dt_all)
        
        grandchildren_positions = np.array([gc['position'] for gc in tree.grandchildren])
        dist_matrix = pairwise_sqdist(grandchildren_positions)
        
        # Создаем маску нелегитимных пар
        mask = np.full_like(dist_matrix, np.inf)
        for gc_idx, allowed_partners in pairing_map.items():
            mask[gc_idx, allowed_partners] = 1.0
            
        # Применяем маску: расстояние до нелегитимных пар становится бесконечным
        masked_dist_matrix = dist_matrix * mask
        
        # Суммируем только конечные значения (верхний треугольник)
        loss = np.sum(np.triu(masked_dist_matrix, k=1)[np.isfinite(np.triu(masked_dist_matrix, k=1))])

        return loss, dist_matrix

    def callback_function(current_dt_grandchildren):
        fixed_dt_children = spore_tree_config.get_default_dt_vector()[:num_children]
        current_loss, _ = grandchildren_pairing_loss(
            current_dt_grandchildren, 
            fixed_dt_children, 
            tree.pairing_candidate_map
        )
        logging.info(f"Iter {opt_state.iteration}: Loss={current_loss:.6f}")
        opt_state.history.append({'iteration': opt_state.iteration, 'loss': current_loss})
        opt_state.iteration += 1

    logging.info("Определены функции для оптимизации встречи внуков с учетом родственных связей.")

    # --- Запуск оптимизации ---
    fixed_dt_children = spore_tree_config.get_default_dt_vector()[:num_children]
    initial_dt_grandchildren = spore_tree_config.get_default_dt_vector()[num_children:]
    np.random.seed(42)
    initial_dt_grandchildren += np.random.uniform(-0.001, 0.001, size=initial_dt_grandchildren.shape)

    bounds = [(0.001, 0.2)] * num_params
    
    # ---- ВАЖНО: Убедимся, что карта создана до первого вызова objective -----
    evaluator._build_if_needed(np.concatenate([fixed_dt_children, initial_dt_grandchildren]))
    # -----------------------------------------------------------------------

    objective_wrapped = lambda dt_gc: grandchildren_pairing_loss(dt_gc, fixed_dt_children, tree.pairing_candidate_map)[0]

    logging.info("--- НАЧАЛО ОПТИМИЗАЦИИ ВНУКОВ ---")
    
    # --- Сохранение и визуализация начального состояния ---
    initial_dt_all = np.concatenate([fixed_dt_children, initial_dt_grandchildren])
    evaluator._build_if_needed(initial_dt_all)
    initial_tree_state = {
        'root': tree.root.copy(),
        'children': [c.copy() for c in tree.children],
        'grandchildren': [gc.copy() for gc in tree.grandchildren]
    }
    
    # Сохраняем начальную матрицу расстояний
    _, initial_dist_matrix = grandchildren_pairing_loss(
        initial_dt_grandchildren, 
        fixed_dt_children, 
        tree.pairing_candidate_map
    )
    initial_dist_matrix_path = os.path.join(run_dir, 'initial_distance_matrix.csv')
    np.savetxt(initial_dist_matrix_path, initial_dist_matrix, delimiter=',', fmt='%.6f')
    logging.info(f"Начальная матрица расстояний сохранена в {initial_dist_matrix_path}")

    fig, ax = plt.subplots(figsize=(12, 11))
    visualize_spore_tree(initial_tree_state, title="Начальное состояние (до оптимизации)", ax=ax)
    initial_state_path = os.path.join(run_dir, 'initial_state.png')
    plt.savefig(initial_state_path)
    plt.close(fig)
    logging.info(f"Начальное состояние сохранено в {initial_state_path}")

    result = minimize(
        objective_wrapped, initial_dt_grandchildren,
        method='Nelder-Mead',
        bounds=bounds, callback=callback_function,
        options={
            'maxiter': 1000,  # Явно задаем низкий предел итераций
            'disp': True, 
            'xatol': 1e-3,  # Увеличиваем допуск (требуем меньшей точности)
            'fatol': 1e-3   # Увеличиваем допуск (требуем меньшей точности)
        }
    )

    logging.info("--- ОПТИМИЗАЦИЯ ВНУКОВ ЗАВЕРШЕНА ---")

    # --- Сохранение и визуализация конечного состояния ---
    final_dt_grandchildren = result.x
    final_dt_all = np.concatenate([fixed_dt_children, final_dt_grandchildren])
    evaluator._build_if_needed(final_dt_all)
    
    # Сохраняем финальную матрицу расстояний
    _, final_dist_matrix = grandchildren_pairing_loss(
        final_dt_grandchildren, 
        fixed_dt_children,
        tree.pairing_candidate_map
    )
    final_dist_matrix_path = os.path.join(run_dir, 'final_distance_matrix.csv')
    np.savetxt(final_dist_matrix_path, final_dist_matrix, delimiter=',', fmt='%.6f')
    logging.info(f"Финальная матрица расстояний сохранена в {final_dist_matrix_path}")

    title = f"Финальное состояние после {result.nit} итераций"
    if not result.success:
        title += " (Оптимизация НЕ УДАЛАСЬ)"
        logging.warning("Оптимизация не была успешной.")

    fig, ax = plt.subplots(figsize=(12, 11))
    visualize_spore_tree(tree, title=title, ax=ax)
    final_state_path = os.path.join(run_dir, 'final_state.png')
    plt.savefig(final_state_path)
    plt.close(fig)
    logging.info(f"Финальное состояние сохранено в {final_state_path}")

    # --- Сохранение метрик ---
    metrics = {
        'success': bool(result.success),
        'message': result.message,
        'final_loss': result.fun,
        'iterations': result.nit,
        'initial_dt_grandchildren': initial_dt_grandchildren.tolist(),
        'final_dt_grandchildren': final_dt_grandchildren.tolist(),
        'history': opt_state.history
    }
    metrics_path = os.path.join(run_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    logging.info(f"Метрики сохранены в {metrics_path}")


if __name__ == '__main__':
    main()
