import pytest
import numpy as np
import sys
import os

# Добавляем корневую директорию проекта в sys.path
# Это нужно, чтобы можно было импортировать модули из src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spore_tree import SporeTree
from src.spore_tree_config import SporeTreeConfig
from src.pendulum import PendulumSystem

@pytest.fixture
def configured_tree() -> SporeTree:
    """
    Фикстура для создания полностью инициализированного дерева (с детьми и внуками).
    """
    pendulum = PendulumSystem()
    config = SporeTreeConfig(show_debug=False) # Отключаем принты для тестов
    
    tree = SporeTree(pendulum, config)
    tree.create_children()
    tree.create_grandchildren()
    
    return tree

def test_candidate_map_creation_and_structure(configured_tree: SporeTree):
    """
    Проверяет, что карта кандидатов создана, имеет правильную структуру и тип.
    """
    tree = configured_tree
    
    # Критерий 1: В классе Tree существует атрибут pairing_candidate_map.
    assert hasattr(tree, 'pairing_candidate_map'), "Атрибут pairing_candidate_map отсутствует"
    
    # Критерий 2: После создания объекта Tree этот атрибут является словарем и не пуст.
    assert isinstance(tree.pairing_candidate_map, dict), "pairing_candidate_map должен быть словарем"
    assert tree.pairing_candidate_map, "pairing_candidate_map не должен быть пустым"

def test_candidate_map_key_count(configured_tree: SporeTree):
    """
    Проверяет, что количество ключей в карте равно общему числу внуков.
    """
    tree = configured_tree
    
    # Критерий 3: Количество ключей в tree.pairing_candidate_map равно 8.
    num_grandchildren = len(tree.grandchildren)
    assert len(tree.pairing_candidate_map.keys()) == num_grandchildren, \
        f"Ожидалось {num_grandchildren} ключей, но получено {len(tree.pairing_candidate_map.keys())}"
    
    # Проверим, что все global_idx внуков присутствуют в ключах
    grandchild_ids = {gc['global_idx'] for gc in tree.grandchildren}
    map_keys = set(tree.pairing_candidate_map.keys())
    assert grandchild_ids == map_keys, "Ключи в карте не соответствуют ID внуков"

def test_candidate_map_value_length(configured_tree: SporeTree):
    """
    Проверяет, что для каждого внука список кандидатов имеет правильную длину.
    """
    tree = configured_tree
    num_grandchildren = len(tree.grandchildren)
    
    # Найдем количество внуков у одного родителя (должно быть 2)
    parent_to_grandchild_count = {}
    for gc in tree.grandchildren:
        parent_idx = gc['parent_idx']
        parent_to_grandchild_count[parent_idx] = parent_to_grandchild_count.get(parent_idx, 0) + 1
    
    # Предполагаем, что у всех родителей одинаковое число внуков
    siblings_count = list(parent_to_grandchild_count.values())[0]
    
    expected_candidates_count = num_grandchildren - siblings_count
    
    # Критерий 4: Для каждого ключа, длина списка-значения равна 6 (8 - 2).
    for grandchild_id, candidates in tree.pairing_candidate_map.items():
        assert len(candidates) == expected_candidates_count, \
            f"Для внука {grandchild_id} ожидалось {expected_candidates_count} кандидатов, но получено {len(candidates)}"

def test_no_siblings_in_candidates(configured_tree: SporeTree):
    """
    Проверяет, что в списке кандидатов для любого внука отсутствуют его братья/сестры.
    """
    tree = configured_tree
    
    # Создадим маппинг: global_idx -> parent_idx для быстрого доступа
    id_to_parent_map = {gc['global_idx']: gc['parent_idx'] for gc in tree.grandchildren}
    
    # Критерий 5: Для любого внука g, в списке его кандидатов отсутствует его "родной брат".
    for grandchild_id, candidate_ids in tree.pairing_candidate_map.items():
        current_parent_id = id_to_parent_map[grandchild_id]
        
        for candidate_id in candidate_ids:
            candidate_parent_id = id_to_parent_map[candidate_id]
            assert current_parent_id != candidate_parent_id, \
                (f"Ошибка логики: внук {grandchild_id} (родитель {current_parent_id}) "
                 f"не должен спариваться с кандидатом {candidate_id} (родитель {candidate_parent_id}), "
                 f"так как у них один родитель.")

