def extract_pairs_from_chronology(chronology, show=False):
    """
    Извлекает пары внуков по умной логике на основе расстояний.
    
    Логика:
    1. Идем по хронологии встреч для каждого внука
    2. Если встречаем внука с расстоянием < 1e-6, берем его в пару и останавливаемся
    3. Если встречаем родителя с расстоянием < 1e-6, останавливаемся и берем лучшего внука из тех, что видели ДО родителя
    4. Если дошли до конца - берем лучшего доступного внука
    
    Args:
        chronology: результат от create_chronological_meetings()
        show: bool - показать процесс формирования пар
        
    Returns:
        list: список пар вида [(gc_i, gc_j, meeting_info), ...]
    """
    if show:
        print("ФОРМИРОВАНИЕ ПАР ПО УМНОЙ ЛОГИКЕ (РАССТОЯНИЯ < 1e-6 + ЛУЧШИЙ ВНУК)")
        print("=" * 70)
    
    pairs = []
    used_grandchildren = set()  # Чтобы избежать дублирования
    
    # Сортируем внуков по номеру для детерминированного результата
    sorted_gc_indices = sorted(chronology.keys())
    
    for gc_idx in sorted_gc_indices:
        # Пропускаем уже использованных внуков
        if gc_idx in used_grandchildren:
            if show:
                print(f"gc_{gc_idx}: уже в паре, пропускаем")
            continue
        
        meetings = chronology[gc_idx]
        
        if show:
            print(f"\nАнализируем gc_{gc_idx}:")
            print(f"  Всего встреч в хронологии: {len(meetings)}")
            print(f"  (встречи только с ЧУЖИМИ родителями - свои исключены в таблицах)")
        
        # Идем по хронологии и применяем умную логику
        selected_meeting = None
        best_grandchild_meeting = None  # Лучший внук, встреченный до сих пор
        stop_reason = None
        
        for i, meeting in enumerate(meetings):
            distance = meeting['distance']
            
            if show:
                print(f"    {i+1}. {meeting['partner']}: расст={distance}, тип={meeting['type']}")
            
            if meeting['type'] == 'grandchild':
                partner_idx = meeting['partner_idx']
                
                # Проверяем что партнер еще не использован
                if partner_idx not in used_grandchildren:
                    # Обновляем лучшего внука
                    if best_grandchild_meeting is None or distance < best_grandchild_meeting['distance']:
                        best_grandchild_meeting = meeting
                        if show:
                            print(f"      📝 Обновили лучшего внука: {meeting['partner']} (расст={distance})")
                    
                    # Если расстояние < 1e-6, сразу берем
                    if distance < 1e-6:
                        selected_meeting = meeting
                        stop_reason = f"нашли внука {meeting['partner']} с расстоянием {distance} < 1e-6"
                        if show:
                            print(f"      ✅ ВЫБРАН СРАЗУ: {stop_reason}")
                        break
                    else:
                        if show:
                            print(f"      ⏩ Внук доступен, но расстояние {distance} >= 1e-6, продолжаем поиск")
                else:
                    if show:
                        print(f"      ❌ Внук недоступен (уже использован)")
                        
            elif meeting['type'] == 'parent':
                # Все родители в хронологии уже ЧУЖИЕ (свои исключены в таблицах)
                if distance < 1e-6:
                    # Останавливаемся и берем лучшего внука из тех, что видели
                    stop_reason = f"встретили чужого родителя {meeting['partner']} с расстоянием {distance} < 1e-6"
                    if best_grandchild_meeting is not None:
                        selected_meeting = best_grandchild_meeting
                        stop_reason += f", берем лучшего внука {best_grandchild_meeting['partner']}"
                    if show:
                        print(f"      🛑 СТОП: {stop_reason}")
                    break
                else:
                    if show:
                        print(f"      ⏩ Чужой родитель {meeting['partner']}, но расстояние {distance} >= 1e-6, продолжаем поиск")
        
        # Если дошли до конца и ничего не выбрали, берем лучшего внука
        if selected_meeting is None and best_grandchild_meeting is not None:
            selected_meeting = best_grandchild_meeting
            stop_reason = f"дошли до конца хронологии, берем лучшего внука {best_grandchild_meeting['partner']}"
            if show:
                print(f"      🏁 КОНЕЦ ХРОНОЛОГИИ: {stop_reason}")
        
        # Обрабатываем результат
        if selected_meeting:
            partner_idx = selected_meeting['partner_idx']
            
            # Создаем пару
            pair = (gc_idx, partner_idx, selected_meeting)
            pairs.append(pair)
            
            # Помечаем обоих как использованных
            used_grandchildren.add(gc_idx)
            used_grandchildren.add(partner_idx)
            
            if show:
                meeting_time = selected_meeting['meeting_time']
                distance = selected_meeting['distance']
                print(f"  🎯 РЕЗУЛЬТАТ: gc_{gc_idx} + gc_{partner_idx}, t={meeting_time}с, расст={distance}")
        else:
            if show:
                if meetings:
                    print(f"  ❌ РЕЗУЛЬТАТ: gc_{gc_idx} не нашел подходящих внуков")
                else:
                    print(f"  ❌ РЕЗУЛЬТАТ: gc_{gc_idx} вообще нет встреч")
    
    if show:
        unpaired_count = len(chronology) - len(used_grandchildren)
        print(f"\nИТОГОВАЯ СТАТИСТИКА:")
        print("=" * 30)
        print(f"  Сформировано пар: {len(pairs)}")
        print(f"  Внуков в парах: {len(used_grandchildren)}")
        print(f"  Внуков без пар: {unpaired_count}")
        
        print(f"\nСПИСОК ПАР:")
        for i, (gc_i, gc_j, meeting) in enumerate(pairs, 1):
            print(f"  {i}. gc_{gc_i} ↔ gc_{gc_j}: t={meeting['meeting_time']}с, расст={meeting['distance']}")
    
    return pairs


def analyze_pairing_quality(pairs, chronology, show=False):
    """
    Анализирует качество сформированных пар.
    
    Args:
        pairs: результат от extract_pairs_from_chronology()
        chronology: исходная хронология
        show: bool - показать анализ
        
    Returns:
        dict: статистика качества пар
    """
    if not pairs:
        return {'total_pairs': 0, 'avg_distance': 0, 'avg_time': 0}
    
    distances = []
    times = []
    
    for gc_i, gc_j, meeting in pairs:
        distances.append(meeting['distance'])
        times.append(meeting['meeting_time'])
    
    stats = {
        'total_pairs': len(pairs),
        'avg_distance': sum(distances) / len(distances),
        'min_distance': min(distances),
        'max_distance': max(distances),
        'avg_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'ultra_close_pairs': sum(1 for d in distances if d < 1e-6),  # Расстояние < 1e-6
        'very_close_pairs': sum(1 for d in distances if d < 1e-5),   # Расстояние < 1e-5
        'close_pairs': sum(1 for d in distances if d < 1e-4)         # Расстояние < 1e-4
    }
    
    if show:
        print("АНАЛИЗ КАЧЕСТВА ПАР:")
        print("=" * 30)
        print(f"Всего пар: {stats['total_pairs']}")
        print(f"\nРасстояния:")
        print(f"  Среднее: {stats['avg_distance']}")
        print(f"  Минимальное: {stats['min_distance']}")
        print(f"  Максимальное: {stats['max_distance']}")
        print(f"\nВремена встреч:")
        print(f"  Среднее: {stats['avg_time']}с")
        print(f"  Минимальное: {stats['min_time']}с")
        print(f"  Максимальное: {stats['max_time']}с")
        print(f"\nКачество сближения:")
        print(f"  Ультра-близкие (< 1e-6): {stats['ultra_close_pairs']}/{stats['total_pairs']}")
        print(f"  Очень близкие (< 1e-5): {stats['very_close_pairs']}/{stats['total_pairs']}")
        print(f"  Близкие (< 1e-4): {stats['close_pairs']}/{stats['total_pairs']}")
    
    return stats

def get_pair_details(pairs, tree, show=False):
    """
    Получает подробную информацию о сформированных парах.
    
    Args:
        pairs: результат от extract_pairs_from_chronology()
        tree: SporeTree объект для получения информации о внуках
        show: bool - показать детальную информацию
        
    Returns:
        list: список с подробной информацией о парах
    """
    detailed_pairs = []
    
    # Вычисляем минимальное расстояние между родителями для справки
    min_parent_distance = _calculate_min_parent_distance(tree, show=False)
    distance_threshold = min_parent_distance / 10.0
    
    for i, (gc_i, gc_j, meeting_info) in enumerate(pairs):
        gc_i_info = tree.grandchildren[gc_i]
        gc_j_info = tree.grandchildren[gc_j]
        
        # Определяем направления времени
        direction_i = "forward" if gc_i_info['dt'] > 0 else "backward"
        direction_j = "forward" if gc_j_info['dt'] > 0 else "backward"
        
        # Подробная информация о паре
        pair_detail = {
            'pair_index': i,
            'gc_i': gc_i,
            'gc_j': gc_j,
            'gc_i_direction': direction_i,
            'gc_j_direction': direction_j,
            'gc_i_parent': gc_i_info['parent_idx'],
            'gc_j_parent': gc_j_info['parent_idx'],
            'meeting_time': meeting_info['meeting_time'],
            'distance': meeting_info['distance'],
            'time_gc_i': meeting_info['time_gc'],
            'time_gc_j': meeting_info['time_partner'],
            'who_waits': meeting_info['who_waits'],
            'same_parent': gc_i_info['parent_idx'] == gc_j_info['parent_idx'],
            'distance_ratio': meeting_info['distance'] / min_parent_distance,  # Отношение к мин. расстоянию родителей
            'passes_distance_check': meeting_info['distance'] < distance_threshold
        }
        detailed_pairs.append(pair_detail)
    
    if show:
        print("ПОДРОБНАЯ ИНФОРМАЦИЯ О ПАРАХ:")
        print("=" * 70)
        print(f"Порог расстояния: {distance_threshold:.5f} (1/10 от мин. расстояния родителей)")
        
        for detail in detailed_pairs:
            print(f"\nПара {detail['pair_index']}: gc_{detail['gc_i']} ↔ gc_{detail['gc_j']}")
            print(f"  Направления: {detail['gc_i_direction']} ↔ {detail['gc_j_direction']}")
            print(f"  Родители: parent_{detail['gc_i_parent']} ↔ parent_{detail['gc_j_parent']}")
            print(f"  Один родитель: {'ДА' if detail['same_parent'] else 'НЕТ'}")
            print(f"  Время встречи: {detail['meeting_time']:.4f}с")
            print(f"  Времена путешествий: gc_{detail['gc_i']}={detail['time_gc_i']:+.4f}с, "
                  f"gc_{detail['gc_j']}={detail['time_gc_j']:+.4f}с")
            print(f"  Кто ждет: {detail['who_waits']}")
            print(f"  Расстояние: {detail['distance']:.5f} (отношение: {detail['distance_ratio']:.2f})")
            print(f"  Проходит проверку: {'ДА' if detail['passes_distance_check'] else 'НЕТ'}")
    
    return detailed_pairs


def analyze_pair_statistics(detailed_pairs, show=False):
    """
    Анализирует статистику сформированных пар.
    
    Args:
        detailed_pairs: результат от get_pair_details()
        show: bool - показать статистику
        
    Returns:
        dict: статистика пар
    """
    if not detailed_pairs:
        return {'total_pairs': 0}
    
    stats = {
        'total_pairs': len(detailed_pairs),
        'same_parent_pairs': 0,
        'different_parent_pairs': 0,
        'forward_forward_pairs': 0,
        'backward_backward_pairs': 0,
        'forward_backward_pairs': 0,
        'avg_meeting_time': 0,
        'min_meeting_time': float('inf'),
        'max_meeting_time': 0,
        'avg_distance': 0,
        'min_distance': float('inf'),
        'max_distance': 0
    }
    
    meeting_times = []
    distances = []
    
    for detail in detailed_pairs:
        # Родители
        if detail['same_parent']:
            stats['same_parent_pairs'] += 1
        else:
            stats['different_parent_pairs'] += 1
        
        # Направления времени
        if detail['gc_i_direction'] == 'forward' and detail['gc_j_direction'] == 'forward':
            stats['forward_forward_pairs'] += 1
        elif detail['gc_i_direction'] == 'backward' and detail['gc_j_direction'] == 'backward':
            stats['backward_backward_pairs'] += 1
        else:
            stats['forward_backward_pairs'] += 1
        
        # Времена и расстояния
        meeting_times.append(detail['meeting_time'])
        distances.append(detail['distance'])
    
    # Вычисляем статистики
    if meeting_times:
        stats['avg_meeting_time'] = sum(meeting_times) / len(meeting_times)
        stats['min_meeting_time'] = min(meeting_times)
        stats['max_meeting_time'] = max(meeting_times)
    
    if distances:
        stats['avg_distance'] = sum(distances) / len(distances)
        stats['min_distance'] = min(distances)
        stats['max_distance'] = max(distances)
    
    if show:
        print("СТАТИСТИКА ПАР:")
        print("=" * 30)
        print(f"Всего пар: {stats['total_pairs']}")
        print(f"\nПо родителям:")
        print(f"  Один родитель: {stats['same_parent_pairs']}")
        print(f"  Разные родители: {stats['different_parent_pairs']}")
        print(f"\nПо направлениям времени:")
        print(f"  Forward + Forward: {stats['forward_forward_pairs']}")
        print(f"  Backward + Backward: {stats['backward_backward_pairs']}")
        print(f"  Forward + Backward: {stats['forward_backward_pairs']}")
        print(f"\nВремена встреч:")
        print(f"  Среднее: {stats['avg_meeting_time']:.4f}с")
        print(f"  Минимальное: {stats['min_meeting_time']:.4f}с")
        print(f"  Максимальное: {stats['max_meeting_time']:.4f}с")
        print(f"\nРасстояния:")
        print(f"  Среднее: {stats['avg_distance']:.5f}")
        print(f"  Минимальное: {stats['min_distance']:.5f}")
        print(f"  Максимальное: {stats['max_distance']:.5f}")
    
    return stats


def export_pairs_to_csv(detailed_pairs, filename="pairs.csv", show=False):
    """
    Экспортирует информацию о парах в CSV файл.
    
    Args:
        detailed_pairs: результат от get_pair_details()
        filename: имя файла для сохранения
        show: bool - показать информацию о сохранении
        
    Returns:
        pandas.DataFrame: таблица пар
    """
    import pandas as pd
    
    df = pd.DataFrame(detailed_pairs)
    df.to_csv(filename, index=False)
    
    if show:
        print(f"Информация о парах сохранена в файл: {filename}")
        print(f"Строк в таблице: {len(df)}")
    
    return df