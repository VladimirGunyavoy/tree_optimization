def create_meeting_chronology(tree, pendulum, dt_bounds=(0.001, 0.1), show=False):
    """
    Создает хронологию всех возможных встреч для каждого внука.
    
    Args:
        tree: SporeTree - объект дерева с созданными внуками
        pendulum: PendulumSystem - объект маятника
        dt_bounds: tuple - границы поиска |dt|
        show: bool - показать хронологию
        
    Returns:
        dict: {
            'chronology': dict - {gc_idx: [список встреч]},
            'gc_gc_tables': dict - таблицы для встреч внук-внук,
            'gc_parent_tables': dict - таблицы для встреч внук-родитель,
            'summary': dict - краткая сводка
        }
    """
    import numpy as np
    import pandas as pd
    
    if not tree._grandchildren_created:
        raise RuntimeError("Сначала создайте внуков через tree.create_grandchildren()")
    
    if show:
        print("Создание хронологии встреч для всех внуков")
        print("=" * 60)
    
    # Импортируем функции для построения таблиц
    from .build_distance_tables import build_grandchild_distance_tables, build_grandchild_parent_distance_tables
    
    # Шаг 1: Строим таблицы для встреч внук-внук
    if show:
        print("\nЭтап 1: Анализ встреч внук-внук")
        print("-" * 40)
    
    gc_gc_tables = build_grandchild_distance_tables(
        tree, pendulum, dt_bounds=dt_bounds, show=show
    )
    
    # Шаг 2: Строим таблицы для встреч внук-родитель  
    if show:
        print("\nЭтап 2: Анализ встреч внук-родитель")
        print("-" * 40)
    
    gc_parent_tables = build_grandchild_parent_distance_tables(
        tree, pendulum, dt_bounds=dt_bounds, show=show
    )
    
    # Шаг 3: Создаем хронологию для каждого внука
    if show:
        print("\nЭтап 3: Создание хронологии встреч")
        print("-" * 40)
    
    chronology = {}
    n_grandchildren = len(tree.grandchildren)
    
    for gc_idx in range(n_grandchildren):
        meetings = []
        gc = tree.grandchildren[gc_idx]
        
        if show:
            direction = "forward" if gc['dt'] > 0 else "backward"
            print(f"\nВнук gc_{gc_idx} ({direction}):")
        
        # Встречи с другими внуками
        for other_gc_idx in range(n_grandchildren):
            if gc_idx == other_gc_idx:
                continue
                
            distance = gc_gc_tables['distance_table'].iloc[gc_idx, other_gc_idx]
            if not np.isnan(distance):
                time_for_gc = gc_gc_tables['time_table_i'].iloc[gc_idx, other_gc_idx]
                time_for_other = gc_gc_tables['time_table_j'].iloc[gc_idx, other_gc_idx]
                
                meeting = {
                    'type': 'grandchild',
                    'partner': f"gc_{other_gc_idx}",
                    'partner_idx': other_gc_idx,
                    'distance': distance,
                    'time_for_gc': time_for_gc,
                    'time_for_partner': time_for_other,
                    'meeting_quality': 1.0 / (distance + 1e-8),  # Чем меньше расстояние, тем лучше
                    'convergence_velocity': gc_gc_tables['convergence_table'].iloc[gc_idx, other_gc_idx]
                }
                meetings.append(meeting)
        
        # Встречи с чужими родителями
        for parent_idx in range(len(tree.children)):
            if parent_idx == gc['parent_idx']:  # Пропускаем своего родителя
                continue
                
            distance = gc_parent_tables['distance_table'].iloc[gc_idx, parent_idx]
            if not np.isnan(distance):
                time_for_gc = gc_parent_tables['time_table'].iloc[gc_idx, parent_idx]
                
                meeting = {
                    'type': 'parent',
                    'partner': f"parent_{parent_idx}",
                    'partner_idx': parent_idx,
                    'distance': distance,
                    'time_for_gc': time_for_gc,
                    'time_for_partner': None,  # Родитель не двигается
                    'meeting_quality': 1.0 / (distance + 1e-8),
                    'convergence_velocity': gc_parent_tables['convergence_table'].iloc[gc_idx, parent_idx]
                }
                meetings.append(meeting)
        
        # Сортируем встречи по качеству (лучшие первыми)
        meetings.sort(key=lambda x: x['meeting_quality'], reverse=True)
        
        chronology[gc_idx] = meetings
        
        if show:
            if meetings:
                print(f"  Найдено {len(meetings)} возможных встреч:")
                for i, meeting in enumerate(meetings[:5]):  # Показываем топ-5
                    time_info = f"t={meeting['time_for_gc']:+.4f}с"
                    if meeting['time_for_partner'] is not None:
                        time_info += f" (партнер: {meeting['time_for_partner']:+.4f}с)"
                    
                    print(f"    {i+1}. {meeting['partner']}: "
                          f"расст={meeting['distance']:.5f}, {time_info}")
                
                if len(meetings) > 5:
                    print(f"    ... и еще {len(meetings) - 5} встреч")
            else:
                print(f"  Встреч не найдено")
    
    # Шаг 4: Создаем сводку
    summary = {
        'total_grandchildren': n_grandchildren,
        'total_gc_gc_meetings': sum(len([m for m in meetings if m['type'] == 'grandchild']) 
                                   for meetings in chronology.values()) // 2,  # Делим на 2 т.к. считаем дважды
        'total_gc_parent_meetings': sum(len([m for m in meetings if m['type'] == 'parent']) 
                                       for meetings in chronology.values()),
        'grandchildren_with_meetings': sum(1 for meetings in chronology.values() if meetings),
        'best_meetings_per_gc': {}
    }
    
    # Находим лучшую встречу для каждого внука
    for gc_idx, meetings in chronology.items():
        if meetings:
            best = meetings[0]  # Уже отсортированы по качеству
            summary['best_meetings_per_gc'][gc_idx] = {
                'partner': best['partner'],
                'distance': best['distance'],
                'time': best['time_for_gc'],
                'quality': best['meeting_quality']
            }
    
    if show:
        print(f"\nИтоговая сводка:")
        print("=" * 30)
        print(f"Всего внуков: {summary['total_grandchildren']}")
        print(f"Встреч внук-внук: {summary['total_gc_gc_meetings']}")
        print(f"Встреч внук-родитель: {summary['total_gc_parent_meetings']}")
        print(f"Внуков с возможными встречами: {summary['grandchildren_with_meetings']}")
        
        print(f"\nЛучшие встречи:")
        for gc_idx, best in summary['best_meetings_per_gc'].items():
            print(f"  gc_{gc_idx} → {best['partner']}: "
                  f"расст={best['distance']:.5f}, t={best['time']:+.4f}с")
    
    return {
        'chronology': chronology,
        'gc_gc_tables': gc_gc_tables,
        'gc_parent_tables': gc_parent_tables,
        'summary': summary
    }


def export_chronology_to_csv(chronology_result, output_dir="results", show=False):
    """
    Экспортирует результаты хронологии в CSV файлы.
    
    Args:
        chronology_result: результат от create_meeting_chronology()
        output_dir: str - директория для сохранения файлов
        show: bool - показать информацию о сохранении
        
    Returns:
        dict: пути к сохраненным файлам
    """
    import os
    import pandas as pd
    
    # Создаем директорию если не существует
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    
    # 1. Сводная таблица лучших встреч
    best_meetings_data = []
    for gc_idx, meetings in chronology_result['chronology'].items():
        if meetings:
            best = meetings[0]
            best_meetings_data.append({
                'grandchild': f"gc_{gc_idx}",
                'best_partner': best['partner'],
                'partner_type': best['type'],
                'distance': best['distance'],
                'time_for_gc': best['time_for_gc'],
                'time_for_partner': best['time_for_partner'],
                'quality': best['meeting_quality'],
                'convergence_velocity': best['convergence_velocity']
            })
    
    best_df = pd.DataFrame(best_meetings_data)
    best_file = os.path.join(output_dir, "best_meetings.csv")
    best_df.to_csv(best_file, index=False)
    saved_files['best_meetings'] = best_file
    
    # 2. Полная хронология
    full_chronology_data = []
    for gc_idx, meetings in chronology_result['chronology'].items():
        for rank, meeting in enumerate(meetings, 1):
            full_chronology_data.append({
                'grandchild': f"gc_{gc_idx}",
                'rank': rank,
                'partner': meeting['partner'],
                'partner_type': meeting['type'],
                'distance': meeting['distance'],
                'time_for_gc': meeting['time_for_gc'],
                'time_for_partner': meeting['time_for_partner'],
                'quality': meeting['meeting_quality'],
                'convergence_velocity': meeting['convergence_velocity']
            })
    
    full_df = pd.DataFrame(full_chronology_data)
    full_file = os.path.join(output_dir, "full_chronology.csv")
    full_df.to_csv(full_file, index=False)
    saved_files['full_chronology'] = full_file
    
    # 3. Таблицы расстояний
    gc_gc_dist_file = os.path.join(output_dir, "gc_gc_distances.csv")
    chronology_result['gc_gc_tables']['distance_table'].to_csv(gc_gc_dist_file)
    saved_files['gc_gc_distances'] = gc_gc_dist_file
    
    gc_parent_dist_file = os.path.join(output_dir, "gc_parent_distances.csv")
    chronology_result['gc_parent_tables']['distance_table'].to_csv(gc_parent_dist_file)
    saved_files['gc_parent_distances'] = gc_parent_dist_file
    
    if show:
        print(f"Результаты сохранены в директории: {output_dir}")
        for name, path in saved_files.items():
            print(f"  {name}: {path}")
    
    return saved_files


def get_meeting_recommendations(chronology_result, max_recommendations=3, show=False):
    """
    Выдает рекомендации по встречам на основе хронологии.
    
    Args:
        chronology_result: результат от create_meeting_chronology()
        max_recommendations: int - максимальное количество рекомендаций
        show: bool - показать рекомендации
        
    Returns:
        list: список рекомендаций
    """
    recommendations = []
    
    # Собираем все встречи внук-внук
    gc_gc_meetings = []
    for gc_idx, meetings in chronology_result['chronology'].items():
        for meeting in meetings:
            if meeting['type'] == 'grandchild':
                # Добавляем только если еще не добавили обратную пару
                pair_key = tuple(sorted([gc_idx, meeting['partner_idx']]))
                if not any(tuple(sorted([r['gc_i'], r['gc_j']])) == pair_key for r in gc_gc_meetings):
                    gc_gc_meetings.append({
                        'gc_i': gc_idx,
                        'gc_j': meeting['partner_idx'],
                        'distance': meeting['distance'],
                        'time_i': meeting['time_for_gc'],
                        'time_j': meeting['time_for_partner'],
                        'quality': meeting['meeting_quality']
                    })
    
    # Сортируем по качеству и берем топ
    gc_gc_meetings.sort(key=lambda x: x['quality'], reverse=True)
    recommendations.extend(gc_gc_meetings[:max_recommendations])
    
    if show:
        print(f"Топ-{max_recommendations} рекомендаций по встречам:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. gc_{rec['gc_i']} ↔ gc_{rec['gc_j']}: "
                  f"расстояние={rec['distance']:.5f}, "
                  f"времена=({rec['time_i']:+.4f}, {rec['time_j']:+.4f})с")
    
    return recommendations