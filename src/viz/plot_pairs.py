import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

def plot_points_and_arrows(
    X: np.ndarray,
    P: np.ndarray,
    starts: np.ndarray = None,
    title: str = "Soft Matching Result",
    save_dir: str = None,
    metrics: dict = None
):
    """
    Визуализирует конечные точки, начальные точки (если есть) и связи между ними.

    Args:
        X (np.ndarray): Конечные координаты точек (N, d).
        P (np.ndarray): Матрица мягкого соответствия (N, N) от Sinkhorn.
        starts (np.ndarray, optional): Начальные координаты точек (N, d).
        title (str, optional): Заголовок графика.
        save_dir (str, optional): Директория для сохранения. Если None, не сохраняет.
        metrics (dict, optional): Словарь с метриками для сохранения в JSON.
    """
    N = X.shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Рисуем начальные точки и стрелки к конечным, если они есть
    if starts is not None:
        ax.scatter(starts[:, 0], starts[:, 1], c='gray', marker='x', label='Start Positions', s=100, alpha=0.7)
        for i in range(N):
            ax.arrow(
                starts[i, 0], starts[i, 1],
                X[i, 0] - starts[i, 0], X[i, 1] - starts[i, 1],
                color='gray', linestyle='--', alpha=0.5, head_width=0.005
            )

    # 2. Рисуем конечные точки
    ax.scatter(X[:, 0], X[:, 1], c=np.arange(N), cmap='viridis', s=150, zorder=3, label='End Positions')
    for i in range(N):
        ax.text(X[i, 0] + 0.01, X[i, 1] + 0.01, str(i), fontsize=12, zorder=4)

    # 3. Рисуем стрелки наиболее вероятных пар
    matched_pairs = set()
    for i in range(N):
        j = np.argmax(P[i, :])
        # Чтобы не рисовать стрелки в обе стороны для одной пары
        if i != j and tuple(sorted((i, j))) not in matched_pairs:
            ax.arrow(
                X[i, 0], X[i, 1],
                X[j, 0] - X[i, 0], X[j, 1] - X[i, 1],
                color='red', alpha=0.6, head_width=0.01,
                zorder=2, linestyle='-'
            )
            matched_pairs.add(tuple(sorted((i, j))))

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Coordinate 1")
    ax.set_ylabel("Coordinate 2")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    
    # 4. Сохранение
    if save_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(save_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        
        # Сохраняем график
        img_path = os.path.join(run_dir, "final_state.png")
        plt.savefig(img_path, dpi=300)
        print(f"✅ График сохранен в: {img_path}")
        
        # Сохраняем метрики
        if metrics:
            metrics_path = os.path.join(run_dir, "metrics.json")
            # Преобразуем numpy в list для JSON-сериализации
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    metrics[key] = value.tolist()
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"✅ Метрики сохранены в: {metrics_path}")

    plt.show()
