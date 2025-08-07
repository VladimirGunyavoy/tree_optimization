# 📋 ТЗ: Оптимизация системы деревьев спор

## 🎯 **Цель проекта**
Кардинально ускорить систему оптимизации параметров dt для максимизации площади четырехугольника, построенного из средних точек пар внуков в дереве спор.

**Текущая проблема:** Оптимизация 12 параметров dt занимает ~1000 итераций × 12 интеграций = 12,000+ вычислений системы маятника.

**Целевое ускорение:** 10-50x (с десятков секунд до долей секунды).

## 🔧 **Основная функция оптимизации**

```python
def optimize_dt(initial_position, pendulum, config=None):
    """
    Оптимизированная функция поиска оптимальных dt с централизованной конфигурацией.
    
    Args:
        initial_position: np.array([theta, theta_dot]) - начальная позиция
        pendulum: PendulumSystem - объект маятника
        config: dict - конфигурационный словарь (если None, использует defaults)
        
    Returns:
        результат оптимизации + топология дерева + использованный конфиг
        
    Jupyter отладка (управляется через config["debug"]):
        🚀 === ОПТИМИЗАЦИЯ dt v2.0 ===
        📍 Позиция: [3.14, 0.0]
        ⚙️ Конфиг: method={config["optimizer"]["method"]}, ε={config["epsilon"]}
        
        🌱 Этап 1: Создание топологии...
        [детальный вывод create_tree_topology если show_topology_creation=True]
        
        🎯 Этап 2: Оптимизация 12 параметров...
        📊 Метод: {config["optimizer"]["method"]}
        📊 Начальная площадь: 0.123
        📈 Итерация 1/50: площадь=0.145 (улучшение +0.022)
        ...
        ✅ Сходимость за 23 итерации
        
        📋 === РЕЗУЛЬТАТЫ ===
        🏆 Финальная площадь: 0.678 (улучшение +0.555)  
        📏 Расстояния пар: [0.0005, 0.0008, 0.0003, 0.0009]
        ⏱️  Время: 0.15s (было ~30s, ускорение 200x)
        🔄 Итерации: 23 (было ~1000)
    """
```

### **Создание и управление конфигурацией**

```python
def create_config(preset="default", **overrides):
    """
    Создает конфигурацию с возможностью переопределения параметров.
    
    Args:
        preset: str - предустановка ("default", "debug", "performance", "research")
        **overrides - параметры для переопределения
    
    Returns:
        dict - готовая конфигурация
        
    Примеры:
        # Базовая конфигурация
        config = create_config()
        
        # Отладочный режим с переопределением
        config = create_config("debug", epsilon=1e-4)
        
        # Кастомизация отдельных параметров
        config = create_config("performance")
        config["optimizer"]["method"] = "differential_evolution"
        config["debug"]["show_optimization"] = True
    """

def validate_config(config):
    """
    Проверяет корректность конфигурации и выдает предупреждения.
    
    Проверки:
        ✅ Все обязательные поля присутствуют
        ✅ Типы данных корректны
        ✅ Значения в допустимых диапазонах
        ✅ Совместимость настроек между собой
        
    При ошибках выдает детальные сообщения с предложениями исправлений.
    """
```

## 📓 **Поддержка Jupyter Notebook**

### **Интерактивные возможности**

**1. Пошаговое исследование:**
```python
# Создание и анализ топологии
topology = create_tree_topology(initial_pos, pendulum, show=True)

# Тестирование разных dt векторов
test_dt = np.random.uniform(0.01, 0.1, 12)
positions = calculate_grandchildren_positions(topology, test_dt, show=True)
metrics = calculate_metrics(positions, topology['grandchild_pairs'], show=True)

print(f"🎯 Тестовая площадь: {metrics['area']:.6f}")
```

**2. Визуализация промежуточных результатов:**
```python
def plot_optimization_progress(results, show_details=True):
    """
    Интерактивные графики для анализа оптимизации.
    
    При show_details=True:
        📊 График сходимости площади
        📏 График расстояний между парами  
        📈 Эволюция dt параметров
        🎯 Визуализация финального дерева
    """
```

**3. Сравнение методов оптимизации:**
```python
# Быстрое сравнение в одной ячейке
methods = ['L-BFGS-B', 'Differential Evolution', 'SLSQP']
results = compare_optimizers(topology, methods, show=True)

# Автоматическая таблица результатов
display_comparison_table(results)
```

### **Отладочные утилиты**

**1. Профилирование производительности:**
```python
from utils.profiling import profile_function

@profile_function(show_in_jupyter=True)
def benchmark_optimization():
    # Автоматически покажет:
    # ⏱️ Время выполнения каждой функции
    # 💾 Использование памяти  
    # 🔥 Самые медленные участки
    # 📊 Сравнение с предыдущими запусками
```

**2. Интерактивные слайдеры:**
```python
from ipywidgets import interact, FloatSlider

@interact(
    dt_scale=FloatSlider(min=0.1, max=2.0, step=0.1, value=1.0),
    epsilon=FloatSlider(min=1e-4, max=1e-2, step=1e-4, value=1e-3)
)
def interactive_optimization(dt_scale, epsilon):
    """Интерактивное тестирование параметров оптимизации."""
    result = optimize_dt(
        initial_pos, pendulum, 
        dt_base=0.1*dt_scale, epsilon=epsilon, 
        show=False, show_progress=True
    )
    
    print(f"🎯 Площадь: {result['final_area']:.6f}")
    print(f"⏱️ Время: {result['optimization_time']:.3f}s")
    
    # Автоматически покажет граф дерева
    visualize_tree(result['tree'])
```

### **Экспорт результатов**

```python
def export_results(result, filename="optimization_results.json", show=True):
    """
    Сохранение результатов с метаданными для воспроизводимости.
    
    При show=True:
        💾 Сохранение в {filename}
        📋 Включено: оптимальные dt, метрики, топология, параметры
        🔗 Ссылка на notebook cell для воспроизведения
        ✅ Результаты сохранены успешно
    """
```

### Текущий процесс:
1. `optimize_dt()` → вызывает `objective_function()` в каждой итерации
2. `objective_function()` → пересчитывает `build_simple_tree()` 
3. `build_simple_tree()` → делает 12 вызовов `pendulum.scipy_rk45_step()`
4. Повторяется 1000+ раз до сходимости

### Структура дерева:
- **Корень:** начальная позиция
- **4 ребенка:** комбинации (u_min/u_max, dt_forward/dt_backward)
- **8 внуков:** по 2 от каждого ребенка с обращенным управлением
- **4 пары внуков:** (0,1), (2,3), (4,5), (6,7) после сортировки

---

## 🚀 **Быстрый старт с конфигурационной системой**

### **Минимальный пример:**
```python
from optimization.optimized_dt import optimize_dt, create_config

# Самый простой способ - дефолтный конфиг
result = optimize_dt(initial_pos, pendulum)

# С готовой предустановкой
config = create_config("debug")  
result = optimize_dt(initial_pos, pendulum, config=config)
```

### **Типичные сценарии:**

**1. Быстрая отладка (вижу все промежуточные результаты):**
```python
config = create_config("debug") 
result = optimize_dt(initial_pos, pendulum, config=config)
```

**2. Максимальная производительность (минимум вывода):**
```python
config = create_config("performance")
result = optimize_dt(initial_pos, pendulum, config=config)
```

**3. Исследование (графики + детали + сохранение):**
```python
config = create_config("research")
result = optimize_dt(initial_pos, pendulum, config=config)
```

**4. Кастомизация (меняю только нужное):**
```python
config = create_config("performance")  # Базовая производительность
config["optimizer"]["method"] = "differential_evolution"  # Но другой метод
config["debug"]["show_progress"] = True  # И прогресс-бар
result = optimize_dt(initial_pos, pendulum, config=config)
```

### **Интерактивное исследование в Jupyter:**
```python
# Создаем виджет для интерактивной настройки
from optimization.jupyter.interactive_widgets import config_widget
widget = config_widget()
widget.display()

# После настройки в виджете:
result = optimize_dt(initial_pos, pendulum, config=widget.get_config())
```

---

## 🚀 **Уровень 1: Структурная оптимизация**

### **1.1 Предварительное построение топологии**

**Задача:** Создать дерево один раз и сохранить его структуру.

```python
def create_tree_topology(initial_position, pendulum, config):
    """
    Создает и сохраняет топологию дерева для быстрого пересчета.
    
    Args:
        initial_position: np.array([theta, theta_dot]) - начальная позиция
        pendulum: PendulumSystem - система маятника
        config: dict - конфигурационный словарь с настройками
    
    Returns:
        tree_topology: dict с полной структурой дерева
    
    Jupyter отладка (при config["debug"]["show_topology_creation"]=True):
        🌱 Создание топологии дерева из позиции [3.14, 0.0]
        📊 Базовый dt: 0.1 (config["dt_base"])
        🍄 Создано 4 ребенка:
           0: forward_max (u=+2.0, dt=+0.1) → [3.12, 0.05]
           1: backward_max (u=+2.0, dt=-0.1) → [3.16, -0.05]
           ...
        👶 Создано 8 внуков после сортировки
        🔄 Порядок обхода: [0, 1, 2, 3, 4, 5, 6, 7]
        📝 Сохранены пары: [(0,1), (2,3), (4,5), (6,7)]
        ✅ Топология готова за 0.05s
    """
```

**Конфигурационные настройки:**
- `config["dt_base"]` - базовый временной шаг
- `config["debug"]["show_topology_creation"]` - показать процесс создания
- `config["performance"]["enable_caching"]` - кэшировать результаты
- `config["export"]["save_topology"]` - сохранять топологию на диск

**Выходные данные:**
```python
tree_topology = {
    'initial_position': np.array([theta, theta_dot]),  # исходная позиция
    'parent_positions': np.array((4, 2)),       # 4 позиции родителей
    'grandchild_pairs': [(0,1), (2,3), (4,5), (6,7)],  # пары для расчета
    'controls': {...},                          # управления для каждого звена
    'traversal_order': [...],                   # порядок обхода внуков
    'dt_mapping': {...},                        # какой dt использует каждое звено
    'creation_time': float,                     # время создания для отладки
    'config_snapshot': {...},                  # копия использованного конфига
    'metadata': {...}                           # дополнительная информация
}
```

### **1.2 Быстрые функции расчета**

**Задача:** Написать оптимизированные функции для целевой функции и ограничений с централизованной конфигурацией.

**1.2.1 Функция расчета позиций внуков**
```python
def calculate_grandchildren_positions(tree_topology, dt_vector, config):
    """
    Быстро пересчитывает позиции 8 внуков по новому вектору dt.
    
    Args:
        tree_topology: предварительно построенная топология
        dt_vector: np.array(12) - новые значения dt
        config: dict - конфигурационный словарь
    
    Returns:
        grandchildren_positions: np.array((8, 2)) - позиции всех внуков
        
    Использует настройки:
        config["debug"]["show_calculations"] - детальный вывод
        config["performance"]["enable_caching"] - кэширование результатов
        config["performance"]["enable_vectorization"] - векторизация вычислений
        
    Jupyter отладка (при config["debug"]["show_calculations"]=True):
        🌱 Пересчет позиций внуков
        📊 dt_vector: [0.1, 0.05, ...]
        🎯 Позиция внука 0: [1.234, 0.567] (родитель 0, dt=0.05)
        ...
        💾 Кэш: 15/20 попаданий (75%)
        ✅ Все 8 внуков пересчитаны за 0.001s
    """
```

**1.2.2 Функция расчета метрик**
```python  
def calculate_metrics(grandchildren_positions, pairs, config):
    """
    Быстро вычисляет все необходимые метрики.
    
    Args:
        grandchildren_positions: np.array((8, 2)) - позиции внуков
        pairs: list - индексы пар для анализа
        config: dict - конфигурационный словарь
    
    Returns:
        {
            'pair_distances': np.array(4),      # расстояния между парами
            'mean_points': np.array((4, 2)),    # средние точки пар  
            'area': float                       # площадь четырехугольника
        }
    
    Использует настройки:
        config["debug"]["show_calculations"] - показать промежуточные расчеты
        config["performance"]["enable_vectorization"] - ускоренные вычисления
        config["visualization"]["auto_plot"] - автоматические графики
        
    Jupyter отладка (при config["debug"]["show_calculations"]=True):
        🔍 Анализ пар внуков
        📏 Пара 0 (внуки 0-1): расстояние = 0.0123
        📍 Средняя точка пары 0: [1.1, 0.3]  
        ...
        📊 Четырехугольник: площадь = 0.456
        ✅ Метрики рассчитаны
    """
```

**1.2.3 Оптимизированная целевая функция**
```python
def fast_objective_function(dt_vector, tree_topology, config):
    """
    Быстрая целевая функция без пересчета всего дерева.
    
    Args:
        dt_vector: np.array(12) - параметры для оптимизации
        tree_topology: предварительно построенная топология
        config: dict - конфигурационный словарь
    
    Returns:
        -area  # отрицательная площадь для минимизации
    
    Использует настройки:
        config["debug"]["show_optimization"] - вывод каждого вызова
        config["profiling"]["enable_timing"] - замеры времени
        config["performance"]["enable_caching"] - кэширование
        
    Jupyter отладка (при config["debug"]["show_optimization"]=True):
        🎯 Вызов целевой функции #1247
        📊 dt: [0.1, 0.05, 0.1, 0.05, ...]
        🧮 Площадь: 0.456
        📉 Objective: -0.456 (минимизируем)
        ⏱️ Время: 0.003s
    """
```

**1.2.4 Оптимизированные ограничения**
```python
def fast_constraint_function(dt_vector, tree_topology, pair_idx, config):
    """
    Быстрая функция ограничений для конкретной пары.
    
    Args:
        dt_vector: np.array(12) - параметры для оптимизации
        tree_topology: предварительно построенная топология  
        pair_idx: int - индекс пары (0, 1, 2, 3)
        config: dict - конфигурационный словарь
    
    Returns:
        epsilon - distance  # > 0 означает выполнение ограничения
    
    Использует настройки:
        config["epsilon"] - допустимое расстояние
        config["debug"]["show_constraints"] - детали проверки
        
    Jupyter отладка (при config["debug"]["show_constraints"]=True):
        🔒 Проверка ограничения пары {pair_idx}
        📏 Расстояние: {distance:.6f}
        📝 Допуск ε: {config["epsilon"]}  
        ✅ Ограничение {"выполнено" if constraint > 0 else "нарушено"}: {constraint:.6f}
    """
```

### **1.3 Ожидаемое ускорение Уровня 1**
- **Текущее:** 12 интеграций × 1000 итераций = 12,000 вычислений  
- **После оптимизации:** 12 интеграций × 1 раз + 1000 быстрых пересчетов ≈ 12 вычислений
- **Ускорение:** ~1000x на этапе построения траекторий

---

## 🔧 **Уровень 2: Алгоритмическая оптимизация**

### **2.1 Замена оптимизатора**

**Текущий проблемы SLSQP:**
- Медленная сходимость на гладких функциях  
- Не использует структуру задачи
- Много итераций для 12D задачи

**Кандидаты на замену:**
1. **L-BFGS-B** - для гладких функций с bounds
2. **Differential Evolution** - глобальный, не нужны градиенты  
3. **Powell** - не нужны градиенты, хорош для средних размерностей

**Требования:**
- Реализовать все 3 метода
- Сравнить производительность на тестовых задачах
- Выбрать лучший как default

### **2.2 Аналитические градиенты (опционально)**

**Задача:** Вычислить производные площади по dt аналитически.

**Цепочка дифференцирования:**
```
∂area/∂dt_i = ∂area/∂mean_points × ∂mean_points/∂positions × ∂positions/∂dt_i
```

**Преимущества:**
- Сходимость за 10-50 итераций вместо 500-1000
- Более стабильная оптимизация
- Точные производные vs численные

---

## ⚡ **Уровень 3: Низкоуровневая оптимизация**  

### **3.1 Векторизация и кэширование**

**Кэш интеграций:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_rk45_step(state_key, control, dt):
    """Кэширование результатов интеграции"""
```

**Векторизация вычислений:**
- Заменить циклы на numpy операции
- Batch обработка нескольких траекторий  
- Оптимизация структур данных (arrays вместо lists)

### **3.2 JIT-компиляция (Numba)**

**Кандидаты для JIT:**
- `calculate_grandchildren_positions()` 
- `calculate_metrics()`
- `fast_objective_function()`

**Ожидаемое ускорение:** 2-10x на численных вычислениях.

---

## 📝 **Технические требования**

### **Входные данные:**
- `initial_position: np.array([theta, theta_dot])` - начальная позиция
- `pendulum: PendulumSystem` - система маятника  
- `config: dict` - конфигурационный словарь с настройками (если None, используются defaults)

### **Выходные данные:**
```python
{
    # === РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ===
    'success': bool,
    'optimal_dt_children': np.array(4),      # оптимальные dt детей
    'optimal_dt_grandchildren': np.array(8), # оптимальные dt внуков  
    'final_area': float,                     # достигнутая площадь
    'final_distances': np.array(4),          # расстояния в парах
    'optimization_time': float,              # время оптимизации
    'iterations': int,                       # количество итераций
    
    # === СТРУКТУРЫ ДАННЫХ ===
    'tree_topology': dict,                   # сохраненная топология
    'optimization_history': list,            # история сходимости
    
    # === КОНФИГУРАЦИЯ И ОТЛАДКА ===
    'config_used': dict,                     # использованная конфигурация
    'debug_info': dict                       # отладочная информация (если включена)
}
```

### **Требования для конфигурационной системы:**
- ✅ **Централизованность** - все настройки в одном месте
- ✅ **Валидация** - автоматическая проверка корректности конфигурации
- ✅ **Предустановки** - готовые конфиги для разных сценариев
- ✅ **Иерархичность** - поддержка вложенных настроек  
- ✅ **Переопределение** - легкая кастомизация отдельных параметров
- ✅ **Обратная совместимость** - старый интерфейс работает (см. ниже)

### **🔄 Обратная совместимость**

Старый интерфейс с отдельными параметрами продолжает работать:

```python
# === СТАРЫЙ ИНТЕРФЕЙС (продолжает работать) ===
result = optimize_dt(
    initial_position=initial_pos,
    pendulum=pendulum, 
    dt_base=0.1,
    epsilon=1e-3, 
    dt_bounds=(0.001, 0.2),
    show=True,
    show_progress=True
)

# === НОВЫЙ ИНТЕРФЕЙС (рекомендуется) ===  
config = create_config("research")
config.update({
    "dt_base": 0.1,
    "epsilon": 1e-3,
    "dt_bounds": (0.001, 0.2),
    "debug": {"show_optimization": True, "show_progress": True}
})
result = optimize_dt(initial_pos, pendulum, config=config)
```

**Автоматическое преобразование:** При использовании старого интерфейса функция автоматически создает конфиг из переданных параметров.

### **📋 Сводная таблица настроек**

| Старый параметр | Новый путь в config | Описание |
|----------------|-------------------|----------|
| `dt_base=0.1` | `config["dt_base"]` | Базовый временной шаг |
| `epsilon=1e-3` | `config["epsilon"]` | Допуск схождения пар |
| `dt_bounds=(0.001, 0.2)` | `config["dt_bounds"]` | Границы поиска dt |
| `show=True` | `config["debug"]["show_optimization"]` | Показать процесс оптимизации |
| `show_progress=True` | `config["debug"]["show_progress"]` | Прогресс-бар |
| - | `config["optimizer"]["method"]` | Метод оптимизации |
| - | `config["performance"]["enable_caching"]` | Кэширование |
| - | `config["visualization"]["auto_plot"]` | Автоматические графики |

### **Ограничения совместимости:**
- Сохранить существующий интерфейс `optimize_dt()`
- Поддержать все текущие параметры
- Обратная совместимость с `visualize_tree()`
- Сохранить точность результатов

---

## 🧪 **План тестирования (после уровня 3)**

### **Функциональные тесты:**
- Сравнение результатов старой и новой версий  
- Проверка выполнения ограничений
- Тестирование граничных случаев

### **Производительные тесты:**
- Сравнение времени выполнения разных версий
- Тестирование разных оптимизаторов
- Профилирование узких мест

### **Метрики успеха:**
- ⚡ **Время оптимизации:** < 1 секунды (цель < 0.1 сек)
- 🎯 **Качество решения:** не хуже текущего  
- 📈 **Итерации до сходимости:** < 100 (цель < 50)
- 💾 **Потребление памяти:** не больше текущего

---

## 📅 **Приоритеты реализации**

### **Высокий приоритет (Уровень 1):**
1. Создание конфигурационной системы (`config/`)
2. Создание `tree_topology` структуры
3. Функция `calculate_grandchildren_positions()`  
4. Функция `calculate_metrics()`
5. Быстрые `objective_function` и `constraint_function`

### **Средний приоритет (Уровень 2):**  
1. Тестирование альтернативных оптимизаторов
2. Базовое кэширование результатов
3. Векторизация вычислений

### **Низкий приоритет (Уровень 3):**
1. Аналитические градиенты
2. JIT-компиляция (Numba)
3. Продвинутое кэширование
4. Бенчмарки и тесты производительности (после завершения оптимизации)

---

## 🔧 **Упрощенная архитектура файлов**

```
optimization/
├── core/
│   ├── tree_topology.py      # Структура дерева + config
│   ├── fast_calculations.py  # Быстрые вычисления + config
│   └── cached_integration.py # Кэширование интеграций
├── optimizers/  
│   ├── lbfgs_optimizer.py    # L-BFGS-B реализация + config
│   └── de_optimizer.py       # Differential Evolution + config 
├── config/
│   ├── __init__.py           # Конфигурационная система
│   ├── defaults.py           # Дефолтные настройки и предустановки
│   └── validation.py         # Валидация конфигурации
└── optimized_dt.py          # Главный интерфейс + config система
```

### **Стандарт функций с конфигурацией:**

```python
def example_function(param1, param2, config):
    """
    Стандартная сигнатура функции с конфигурацией:
    
    Args:
        param1, param2: основные параметры функции
        config: dict - конфигурационный словарь
    
    Использует настройки:
        config["debug"]["show_function_name"] - вывод отладки
        config["performance"]["setting"] - влияет на производительность
        
    При config["debug"]["show_function_name"]=True:
        🎯 [НАЗВАНИЕ ФУНКЦИИ] - краткое описание
        📊 Входные параметры: param1=value1, param2=value2
        📊 Используемые настройки: setting1=val1, setting2=val2
        ⏱️ [опционально] Время начала
        
        🔄 Промежуточные шаги:
           • Шаг 1: описание + результат
           • Шаг 2: описание + результат  
           ...
           
        📋 Итоговые результаты:
           • Ключевая метрика 1: value  
           • Ключевая метрика 2: value
           
        ✅ Завершено за X.XXs
    """
    # Извлечение настроек из конфига  
    show_debug = config.get("debug", {}).get("show_function_name", False)
    performance_setting = config.get("performance", {}).get("setting", "default")
    
    if show_debug:
        print(f"🎯 {example_function.__name__.upper()} - описание функции")
        print(f"📊 Параметры: param1={param1}, param2={param2}")
        print(f"📊 Настройки: performance.setting={performance_setting}")
    
    start_time = time.time() if show_debug else None
    
    # ... основная логика функции ...
    # Логика может использовать настройки из config для оптимизации
    
    if show_debug:
        execution_time = time.time() - start_time
        print(f"✅ Завершено за {execution_time:.3f}s")
    
    return result

# === Вспомогательные функции для работы с конфигурацией ===

def get_config_value(config, path, default=None):
    """
    Безопасное получение значения из конфига по пути.
    
    Args:
        config: dict - конфигурационный словарь
        path: str - путь в точечной нотации "section.subsection.key"
        default - значение по умолчанию
    
    Returns:
        значение или default если путь не найден
    
    Примеры:
        show_debug = get_config_value(config, "debug.show_optimization", False)
        method = get_config_value(config, "optimizer.method", "L-BFGS-B")
    """

def should_show_debug(config, debug_type):
    """
    Проверяет, нужно ли показывать отладочную информацию определенного типа.
    
    Args:
        config: dict - конфигурация
        debug_type: str - тип отладки ("topology_creation", "calculations", etc.)
    
    Returns:
        bool - показывать ли отладку
    """
```

**Точка входа:** Обновить существующую `optimize_dt()` функцию с поддержкой конфигурационной системы при сохранении обратной совместимости.