def find_converging_grandchild_pairs(gc_gc_convergence_df, show=False):
    """
    Находит все пары внуков с отрицательными скоростями сближения.
    
    Args:
        gc_gc_convergence_df: pandas.DataFrame - таблица скоростей сближения внуков
        show: bool - показать найденные пары
        
    Returns:
        list: список словарей с парами {'gc_i': int, 'gc_j': int, 'velocity': float, 'pair_name': str}
    """
    import numpy as np
    
    converging_pairs = []
    n = len(gc_gc_convergence_df)
    
    # Проходим только верхний треугольник (избегаем дублирования)
    for i in range(n):
        for j in range(i+1, n):
            velocity = gc_gc_convergence_df.iloc[i, j]
            
            if velocity < -1e-6:  # Сближаются (отрицательная производная расстояния)
                converging_pairs.append({
                    'gc_i': i,
                    'gc_j': j,
                    'velocity': velocity,
                    'pair_name': f"gc_{i}-gc_{j}"
                })
    
    # Сортируем по скорости сближения (самые быстро сближающиеся первыми)
    converging_pairs.sort(key=lambda x: x['velocity'])
    
    if show:
        print(f"Найдено {len(converging_pairs)} сближающихся пар внуков:")
        for pair in converging_pairs:
            print(f"  {pair['pair_name']}: скорость сближения = {pair['velocity']:.6f}")
    
    return converging_pairs


def find_converging_grandchild_parent_pairs(gc_parent_convergence_df, show=False):
    """
    Находит все пары внук-родитель с отрицательными скоростями сближения.
    
    Args:
        gc_parent_convergence_df: pandas.DataFrame - таблица сближения внуков с родителями
        show: bool - показать найденные пары
        
    Returns:
        list: список словарей с парами {'gc_idx': int, 'parent_idx': int, 'velocity': float, 'pair_name': str}
    """
    import numpy as np
    
    converging_pairs = []
    
    for gc_idx in range(len(gc_parent_convergence_df)):
        for parent_idx in range(len(gc_parent_convergence_df.columns)):
            velocity = gc_parent_convergence_df.iloc[gc_idx, parent_idx]
            
            # Пропускаем NaN (свой родитель) и положительные скорости
            if not np.isnan(velocity) and velocity < -1e-6:
                converging_pairs.append({
                    'gc_idx': gc_idx,
                    'parent_idx': parent_idx,
                    'velocity': velocity,
                    'pair_name': f"gc_{gc_idx}-parent_{parent_idx}"
                })
    
    # Сортируем по скорости сближения
    converging_pairs.sort(key=lambda x: x['velocity'])
    
    if show:
        print(f"Найдено {len(converging_pairs)} сближающихся пар внук-родитель:")
        for pair in converging_pairs:
            print(f"  {pair['pair_name']}: скорость сближения = {pair['velocity']:.6f}")
    
    return converging_pairs