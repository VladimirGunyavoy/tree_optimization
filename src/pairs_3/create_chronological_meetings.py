def create_chronological_meetings(tree, pendulum, dt_bounds=(0.001, 0.1), show=False):
    """
    Создает хронологию встреч для каждого внука, упорядоченную по времени встречи.
    
    Args:
        tree: SporeTree - объект дерева с созданными внуками
        pendulum: PendulumSystem - объект маятника
        dt_bounds: tuple - границы поиска |dt|
        show: bool - показать хронологию
        
    Returns:
        dict: {gc_idx: [список встреч в хронологическом порядке]}
    """
    import numpy as np
    
    if not tree._grandchildren_created:
        raise RuntimeError("Сначала создайте внуков через tree.create_grandchildren()")
    
    if show:
        print("СОЗДАНИЕ ХРОНОЛОГИИ ВСТРЕЧ ПО ВРЕМЕНИ")
        print("=" * 60)
    
    # Импортируем функции для построения таблиц
    from .build_distance_tables import build_grandchild_distance_tables, build_grandchild_parent_distance_tables
    
    # Строим все таблицы встреч
    if show:
        print("Строим таблицы встреч...")
    
    gc_gc_tables = build_grandchild_distance_tables(tree, pendulum, dt_bounds=dt_bounds, show=False)
    gc_parent_tables = build_grandchild_parent_distance_tables(tree, pendulum, dt_bounds=dt_bounds, show=False)
    
    # Создаем хронологию для каждого внука
    chronology = {}
    
    for gc_idx in range(len(tree.grandchildren)):
        meetings = []
        gc = tree.grandchildren[gc_idx]
        
        if show:
            direction = "forward" if gc['dt'] > 0 else "backward"
            print(f"\nВнук gc_{gc_idx} ({direction}):")
        
        # Собираем встречи с другими внуками
        for other_gc_idx in range(len(tree.grandchildren)):
            if gc_idx == other_gc_idx:
                continue
                
            distance = gc_gc_tables['distance_table'].iloc[gc_idx, other_gc_idx]
            if not np.isnan(distance):
                time_i = gc_gc_tables['time_table_i'].iloc[gc_idx, other_gc_idx]
                time_j = gc_gc_tables['time_table_j'].iloc[gc_idx, other_gc_idx]
                
                # Время встречи = максимум из двух времен (кто-то ждет)
                meeting_time = max(abs(time_i), abs(time_j))
                
                meeting = {
                    'type': 'grandchild',
                    'partner': f"gc_{other_gc_idx}",
                    'partner_idx': other_gc_idx,
                    'distance': distance,
                    'time_gc': time_i,  # время для текущего внука
                    'time_partner': time_j,  # время для партнера
                    'meeting_time': meeting_time,  # время встречи (максимум)
                    'who_waits': 'gc' if abs(time_i) > abs(time_j) else 'partner'
                }
                meetings.append(meeting)
        
        # Собираем встречи с чужими родителями
        for parent_idx in range(len(tree.children)):
            if parent_idx == gc['parent_idx']:  # Пропускаем своего родителя
                continue
                
            distance = gc_parent_tables['distance_table'].iloc[gc_idx, parent_idx]
            if not np.isnan(distance):
                time_gc = gc_parent_tables['time_table'].iloc[gc_idx, parent_idx]
                
                meeting = {
                    'type': 'parent',
                    'partner': f"parent_{parent_idx}",
                    'partner_idx': parent_idx,
                    'distance': distance,
                    'time_gc': time_gc,  # время для внука
                    'time_partner': None,  # родитель не движется
                    'meeting_time': abs(time_gc),  # время встречи
                    'who_waits': None  # родитель не ждет
                }
                meetings.append(meeting)
        
        # КЛЮЧЕВОЕ: сортируем все встречи по времени встречи
        meetings.sort(key=lambda x: x['meeting_time'])
        
        chronology[gc_idx] = meetings
        
        if show:
            if meetings:
                print(f"  Хронология встреч (по времени):")
                for i, meeting in enumerate(meetings):
                    if meeting['type'] == 'grandchild':
                        wait_info = f"(ждет {meeting['who_waits']})" if meeting['who_waits'] else ""
                        print(f"    {i+1}. t={meeting['meeting_time']:.6f}с: {meeting['partner']} "
                              f"[gc: {meeting['time_gc']:+.6f}с, партнер: {meeting['time_partner']:+.6f}с] "
                              f"расст={meeting['distance']} {wait_info}")
                    else:
                        print(f"    {i+1}. t={meeting['meeting_time']:.6f}с: {meeting['partner']} "
                              f"[gc: {meeting['time_gc']:+.6f}с] "
                              f"расст={meeting['distance']}")
            else:
                print(f"  Встреч не найдено")
    
    if show:
        # Общая статистика
        total_gc_meetings = sum(len([m for m in meetings if m['type'] == 'grandchild']) 
                               for meetings in chronology.values()) // 2  # Избегаем двойного подсчета
        total_parent_meetings = sum(len([m for m in meetings if m['type'] == 'parent']) 
                                   for meetings in chronology.values())
        
        print(f"\nОБЩАЯ СТАТИСТИКА:")
        print(f"  Всего внуков: {len(tree.grandchildren)}")
        print(f"  Уникальных встреч внук-внук: {total_gc_meetings}")
        print(f"  Встреч внук-родитель: {total_parent_meetings}")
        print(f"  Внуков с встречами: {sum(1 for meetings in chronology.values() if meetings)}")
    
    return chronology


def export_chronology_to_csv(chronology, filename="chronology.csv", show=False):
    """
    Экспортирует хронологию в CSV файл.
    
    Args:
        chronology: результат от create_chronological_meetings()
        filename: имя файла для сохранения
        show: показать информацию о сохранении
        
    Returns:
        pandas.DataFrame: таблица хронологии
    """
    import pandas as pd
    
    rows = []
    
    for gc_idx, meetings in chronology.items():
        for rank, meeting in enumerate(meetings, 1):
            if meeting['type'] == 'grandchild':
                row = {
                    'grandchild': f"gc_{gc_idx}",
                    'rank': rank,
                    'meeting_time': meeting['meeting_time'],
                    'partner': meeting['partner'],
                    'partner_type': meeting['type'],
                    'distance': meeting['distance'],
                    'time_gc': meeting['time_gc'],
                    'time_partner': meeting['time_partner'],
                    'who_waits': meeting['who_waits']
                }
            else:  # parent
                row = {
                    'grandchild': f"gc_{gc_idx}",
                    'rank': rank,
                    'meeting_time': meeting['meeting_time'],
                    'partner': meeting['partner'],
                    'partner_type': meeting['type'],
                    'distance': meeting['distance'],
                    'time_gc': meeting['time_gc'],
                    'time_partner': None,
                    'who_waits': None
                }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    
    if show:
        print(f"Хронология сохранена в файл: {filename}")
        print(f"Строк в таблице: {len(df)}")
    
    return df


def get_earliest_meetings(chronology, show=False):
    """
    Находит самые ранние встречи для каждого внука.
    
    Args:
        chronology: результат от create_chronological_meetings()
        show: показать результаты
        
    Returns:
        dict: {gc_idx: earliest_meeting}
    """
    earliest_meetings = {}
    
    for gc_idx, meetings in chronology.items():
        if meetings:
            earliest_meetings[gc_idx] = meetings[0]  # Уже отсортированы по времени
    
    if show:
        print("САМЫЕ РАННИЕ ВСТРЕЧИ:")
        print("-" * 40)
        for gc_idx, meeting in earliest_meetings.items():
            if meeting['type'] == 'grandchild':
                print(f"gc_{gc_idx}: t={meeting['meeting_time']:.4f}с → {meeting['partner']} "
                      f"(расст={meeting['distance']:.5f})")
            else:
                print(f"gc_{gc_idx}: t={meeting['meeting_time']:.4f}с → {meeting['partner']} "
                      f"(расст={meeting['distance']:.5f})")
    
    return earliest_meetings


def analyze_meeting_patterns(chronology, show=False):
    """
    Анализирует паттерны встреч в хронологии.
    
    Args:
        chronology: результат от create_chronological_meetings()
        show: показать анализ
        
    Returns:
        dict: статистика паттернов
    """
    stats = {
        'total_grandchildren': len(chronology),
        'gc_with_meetings': 0,
        'avg_meetings_per_gc': 0,
        'earliest_meeting_time': float('inf'),
        'latest_meeting_time': 0,
        'gc_gc_meetings': 0,
        'gc_parent_meetings': 0
    }
    
    all_meeting_times = []
    total_meetings = 0
    
    for gc_idx, meetings in chronology.items():
        if meetings:
            stats['gc_with_meetings'] += 1
            total_meetings += len(meetings)
            
            # Самые ранние и поздние встречи
            earliest = meetings[0]['meeting_time']
            latest = meetings[-1]['meeting_time']
            
            stats['earliest_meeting_time'] = min(stats['earliest_meeting_time'], earliest)
            stats['latest_meeting_time'] = max(stats['latest_meeting_time'], latest)
            
            # Типы встреч
            for meeting in meetings:
                all_meeting_times.append(meeting['meeting_time'])
                if meeting['type'] == 'grandchild':
                    stats['gc_gc_meetings'] += 1
                else:
                    stats['gc_parent_meetings'] += 1
    
    # Исправляем двойной подсчет встреч внук-внук
    stats['gc_gc_meetings'] //= 2
    
    stats['avg_meetings_per_gc'] = total_meetings / stats['total_grandchildren'] if stats['total_grandchildren'] > 0 else 0
    
    if show:
        print("АНАЛИЗ ПАТТЕРНОВ ВСТРЕЧ:")
        print("-" * 30)
        print(f"Всего внуков: {stats['total_grandchildren']}")
        print(f"Внуков с встречами: {stats['gc_with_meetings']}")
        print(f"Среднее встреч на внука: {stats['avg_meetings_per_gc']:.1f}")
        print(f"Самая ранняя встреча: {stats['earliest_meeting_time']:.4f}с")
        print(f"Самая поздняя встреча: {stats['latest_meeting_time']:.4f}с")
        print(f"Встреч внук-внук: {stats['gc_gc_meetings']}")
        print(f"Встреч внук-родитель: {stats['gc_parent_meetings']}")
    
    return stats