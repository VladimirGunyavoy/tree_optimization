import numpy as np
from scipy.optimize import minimize
from .create_distance_constraints import create_distance_constraints, test_constraints
from .tree_area_evaluator import TreeAreaEvaluator


def optimize_tree_area(tree, pairs, pendulum, constraint_distance=1e-5, 
                      dt_bounds=(0.001, 0.1), max_iterations=1000, 
                      optimization_method='SLSQP', show=False):
    """
    Оптимизирует площадь дерева спор при ограничениях на расстояния между парами.
    
    Решает задачу:
    maximize: площадь дерева
    subject to: расстояния между парами <= constraint_distance
    
    Args:
        tree: исходное дерево SporeTree
        pairs: список пар [(gc_i, gc_j, meeting_info), ...] от find_optimal_pairs()
        pendulum: объект маятника
        constraint_distance: максимально допустимое расстояние в парах
        dt_bounds: границы для всех dt (min_dt, max_dt)
        max_iterations: максимальное количество итераций оптимизации
        optimization_method: метод оптимизации ('SLSQP', 'L-BFGS-B', etc.)
        show: вывод отладочной информации
        
    Returns:
        dict: {
            'success': bool - успех оптимизации,
            'optimized_area': float - итоговая площадь,
            'original_area': float - исходная площадь,
            'improvement': float - улучшение площади,
            'optimized_dt_vector': np.array - оптимальные времена [12],
            'optimized_tree': SporeTree - оптимизированное дерево,
            'optimization_result': scipy result - полный результат scipy,
            'constraint_violations': dict - нарушения констрейнтов
        }
        None при ошибке
    """
    
    try:
        if show:
            print("ОПТИМИЗАЦИЯ ПЛОЩАДИ ДЕРЕВА С КОНСТРЕЙНТАМИ ПАР")
            print("="*60)
        
        # ================================================================
        # ПРОВЕРКИ ВХОДНЫХ ДАННЫХ
        # ================================================================
        
        if not pairs:
            if show:
                print("Ошибка: Список пар пуст")
            return None
            
        if not hasattr(tree, 'grandchildren') or len(tree.grandchildren) == 0:
            if show:
                print("Ошибка: В дереве нет внуков")
            return None
        
        # ================================================================
        # СОЗДАНИЕ JIT-ОПТИМИЗИРОВАННОГО AREA EVALUATOR
        # ================================================================
        
        try:
            # Убеждаемся что исходное дерево имеет детей и внуков
            if not tree._children_created:
                tree.create_children(show=False)
            if not tree._grandchildren_created:
                tree.create_grandchildren(show=False)
            
            # Создаем TreeAreaEvaluator
            area_evaluator = TreeAreaEvaluator(tree, show=show and False)
            
            # Получаем исходный dt_vector
            original_dt_children = np.abs([child['dt'] for child in tree.children])
            original_dt_grandchildren = np.abs([gc['dt'] for gc in tree.grandchildren])
            original_dt_vector = np.hstack([original_dt_children, original_dt_grandchildren])
            
            # Вычисляем исходную площадь
            original_area = area_evaluator.area(original_dt_vector)
            
            if show:
                print(f"TreeAreaEvaluator создан")
                print(f"Исходная площадь дерева: {original_area:.6f}")
                
        except Exception as e:
            if show:
                print(f"Ошибка создания TreeAreaEvaluator: {e}")
            return None
        
        # ================================================================
        # СОЗДАНИЕ КОНСТРЕЙНТОВ
        # ================================================================
        
        if show:
            print("Создание констрейнтов расстояний...")
        
        constraint_functions, constraint_info = create_distance_constraints(
            pairs, tree, pendulum, constraint_distance, show=show and False
        )
        
        if not constraint_functions:
            if show:
                print("Ошибка: Не удалось создать констрейнты")
            return None
        
        # Преобразуем в формат scipy
        scipy_constraints = [{'type': 'ineq', 'fun': func} for func in constraint_functions]
        
        if show:
            print(f"Создано {len(scipy_constraints)} констрейнтов")
        
        # ================================================================
        # ПОДГОТОВКА JIT-ОПТИМИЗИРОВАННОЙ ЦЕЛЕВОЙ ФУНКЦИИ
        # ================================================================
        
        def objective_function(dt_vector):
            """
            JIT-оптимизированная целевая функция: -площадь (минимизируем для максимизации площади).
            
            Использует TreeAreaEvaluator для быстрого пересчета площади без пересоздания дерева.
            
            Args:
                dt_vector: np.array из 12 элементов [4 dt детей + 8 dt внуков]
                
            Returns:
                float: -площадь дерева (для минимизации)
            """
            try:
                # TreeAreaEvaluator ожидает положительные dt и сам применяет знаки
                dt_vector_abs = np.abs(dt_vector)
                
                # Быстрое вычисление площади через JIT-оптимизированный area_evaluator
                area = area_evaluator.area(dt_vector_abs)
                
                # Возвращаем отрицательную площадь для минимизации
                return -area
                
            except Exception as e:
                # При ошибке возвращаем большое положительное число (плохая площадь)
                if show:
                    print(f"Ошибка в целевой функции: {e}")
                return 1e6
        
        # ================================================================
        # НАЧАЛЬНОЕ ПРИБЛИЖЕНИЕ И ГРАНИЦЫ
        # ================================================================
        
        # Начальное приближение: исходные времена дерева (уже вычислены выше)
        x0 = original_dt_vector.copy()
        
        # Границы: все времена положительные
        bounds = [(dt_bounds[0], dt_bounds[1]) for _ in range(12)]
        
        if show:
            print(f"Начальное приближение: {x0}")
            print(f"Границы dt: {dt_bounds}")
            print(f"Метод оптимизации: {optimization_method}")
            print(f"Максимум итераций: {max_iterations}")
        
        # ================================================================
        # ТЕСТИРОВАНИЕ НАЧАЛЬНОГО ПРИБЛИЖЕНИЯ
        # ================================================================
        
        if show:
            print("\nТестирование начального приближения...")
            
            # Тестируем JIT-оптимизированную целевую функцию
            initial_objective = objective_function(x0)
            print(f"Начальная целевая функция: {initial_objective:.6f} (площадь: {-initial_objective:.6f})")
            
            # Проверяем что площади совпадают
            if abs(-initial_objective - original_area) > 1e-10:
                print(f"ВНИМАНИЕ: Несоответствие площади в evaluator!")
                print(f"Original: {original_area:.10f}, Evaluator: {-initial_objective:.10f}")
            else:
                print(f"Проверка evaluator: OK")
            
            # Тестируем констрейнты
            constraint_test = test_constraints(constraint_functions, x0, constraint_info, show=show and False)
            satisfied_count = constraint_test.get('summary', {}).get('satisfied_count', 0)
            total_count = constraint_test.get('summary', {}).get('total_constraints', 0)
            print(f"Начальные констрейнты: {satisfied_count}/{total_count} выполнено")
        
        # ================================================================
        # ОПТИМИЗАЦИЯ
        # ================================================================
        
        if show:
            print(f"\nЗапуск оптимизации...")
        
        # Настройки оптимизации
        options = {
            'maxiter': max_iterations,
            'ftol': 1e-9,
            'disp': show
        }
        
        # Запуск оптимизации
        optimization_result = minimize(
            fun=objective_function,
            x0=x0,
            method=optimization_method,
            bounds=bounds,
            constraints=scipy_constraints,
            options=options
        )
        
        if show:
            print(f"Оптимизация завершена:")
            print(f"  Успех: {optimization_result.success}")
            print(f"  Сообщение: {optimization_result.message}")
            print(f"  Итераций: {optimization_result.get('nit', 'N/A')}")
            print(f"  Вызовов функции: {optimization_result.get('nfev', 'N/A')}")
        
        # ================================================================
        # АНАЛИЗ РЕЗУЛЬТАТА
        # ================================================================
        
        if not optimization_result.success:
            if show:
                print(f"Оптимизация не сошлась: {optimization_result.message}")
            # Возвращаем частичный результат
            return {
                'success': False,
                'optimization_result': optimization_result,
                'error': optimization_result.message
            }
        
        # Извлекаем оптимальный вектор
        optimized_dt_vector = optimization_result.x
        optimized_area = -optimization_result.fun  # Восстанавливаем площадь
        improvement = optimized_area - original_area
        
        if show:
            print(f"\nРезультаты оптимизации:")
            print(f"  Исходная площадь: {original_area:.6f}")
            print(f"  Оптимизированная площадь: {optimized_area:.6f}")
            print(f"  Улучшение: {improvement:.6f} ({improvement/original_area*100:.2f}%)")
        
        # ================================================================
        # СОЗДАНИЕ ОПТИМИЗИРОВАННОГО ДЕРЕВА
        # ================================================================
        
        try:
            # Извлекаем оптимальные времена
            dt_children_opt = np.abs(optimized_dt_vector[0:4])
            dt_grandchildren_opt = np.abs(optimized_dt_vector[4:12])
            
            # Создаем оптимизированное дерево ОДИН РАЗ (не в циклах оптимизации!)
            optimized_tree = tree.__class__(
                pendulum=pendulum,
                config=tree.config,
                dt_children=dt_children_opt,
                dt_grandchildren=dt_grandchildren_opt,
                show=False
            )
            
            if show:
                print(f"Оптимизированное дерево создано")
                
        except Exception as e:
            if show:
                print(f"Ошибка создания оптимизированного дерева: {e}")
            optimized_tree = None
        
        # ================================================================
        # ПРОВЕРКА КОНСТРЕЙНТОВ В ФИНАЛЬНОМ РЕШЕНИИ
        # ================================================================
        
        constraint_violations = test_constraints(
            constraint_functions, optimized_dt_vector, constraint_info, show=show and False
        )
        
        if show:
            violated_count = constraint_violations.get('summary', {}).get('total_constraints', 0) - \
                           constraint_violations.get('summary', {}).get('satisfied_count', 0)
            
            print(f"\nПроверка финальных констрейнтов:")
            print(f"  Нарушено: {violated_count}/{len(constraint_functions)}")
            
            if violated_count > 0:
                print(f"  ВНИМАНИЕ: Есть нарушения констрейнтов!")
                for i, result in constraint_violations.items():
                    if isinstance(result, dict) and not result.get('satisfied', True):
                        gc_i = constraint_info[i]['gc_i']
                        gc_j = constraint_info[i]['gc_j']
                        distance = result['distance']
                        print(f"    Пара gc_{gc_i}-gc_{gc_j}: расстояние {distance:.6f} > {constraint_distance}")
        
        # ================================================================
        # ФИНАЛЬНЫЕ ВРЕМЕНА
        # ================================================================
        
        if show:
            print(f"\nФинальные времена:")
            print(f"  dt_children: {[f'{dt:.6f}' for dt in dt_children_opt]}")
            print(f"  dt_grandchildren: {[f'{dt:.6f}' for dt in dt_grandchildren_opt]}")
        
        return {
            'success': True,
            'optimized_area': optimized_area,
            'original_area': original_area,
            'improvement': improvement,
            'improvement_percent': improvement / original_area * 100,
            'optimized_dt_vector': optimized_dt_vector,
            'optimized_dt_children': dt_children_opt,
            'optimized_dt_grandchildren': dt_grandchildren_opt,
            'optimized_tree': optimized_tree,
            'optimization_result': optimization_result,
            'constraint_violations': constraint_violations,
            'pairs_count': len(pairs),
            'constraints_count': len(constraint_functions)
        }
        
    except Exception as e:
        if show:
            print(f"Критическая ошибка оптимизации: {e}")
        return None


def compare_optimization_results(original_tree, optimization_result, show=False):
    """
    Сравнивает исходное и оптимизированное дерево.
    
    Args:
        original_tree: исходное дерево
        optimization_result: результат от optimize_tree_area()
        show: вывод сравнения
        
    Returns:
        dict: статистика сравнения
    """
    
    if not optimization_result or not optimization_result['success']:
        if show:
            print("Оптимизация не удалась, сравнение невозможно")
        return None
    
    try:
        optimized_tree = optimization_result['optimized_tree']
        
        comparison = {
            'area_improvement': optimization_result['improvement'],
            'area_improvement_percent': optimization_result['improvement_percent'],
            'original_area': optimization_result['original_area'],
            'optimized_area': optimization_result['optimized_area'],
            'constraints_satisfied': optimization_result['constraint_violations']['summary']['all_satisfied']
        }
        
        if show:
            print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ ОПТИМИЗАЦИИ")
            print("="*50)
            print(f"Исходная площадь: {comparison['original_area']:.6f}")
            print(f"Оптимизированная площадь: {comparison['optimized_area']:.6f}")
            print(f"Улучшение: {comparison['area_improvement']:+.6f} ({comparison['area_improvement_percent']:+.2f}%)")
            print(f"Все констрейнты выполнены: {'Да' if comparison['constraints_satisfied'] else 'Нет'}")
            
            print(f"\nПары в оптимизации: {optimization_result['pairs_count']}")
            print(f"Констрейнтов: {optimization_result['constraints_count']}")
        
        return comparison
        
    except Exception as e:
        if show:
            print(f"Ошибка сравнения: {e}")
        return None