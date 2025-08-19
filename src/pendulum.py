import numpy as np
from scipy.linalg import expm
from typing import Tuple
from scipy.integrate import solve_ivp
import numba
from numba import njit, prange, float64

class PendulumSystem:
    """
    Класс, описывающий систему маятника.
    Позволяет выполнять линеаризацию и дискретизацию в произвольном состоянии.
    """
    def __init__(self, 
                 g: float = 9.81,
                 l: float = 2.0,
                 m: float = 1.0,
                 damping: float = 0.1, 
                 max_control: float = 2):
        self.g: float = g
        self.l: float = l
        self.m: float = m
        self.damping: float = damping
        self.max_control: float = float(max_control)
        
        # Оптимизация: кэш для избежания повторных матричных вычислений
        self._linearization_cache = {}  # key: (theta_0,), value: (A_cont, B_cont)
        self._discretization_cache = {}  # key: (A_hash, B_hash, dt), value: (A_d, B_d)

        self._inv_ml2 = 1.0 / (m * l * l)   # часто используется в ядре
        
    def get_control_bounds(self) -> np.ndarray:
        return np.array([-self.max_control, self.max_control])
        
    def get_linearized_matrices_at_state(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Линеаризация нелинейной динамики маятника в произвольном состоянии.
        Возвращает непрерывные матрицы A и B.
        """
        theta_0, _ = state
        
        # Оптимизация: кэширование результатов по округленному theta_0
        cache_key = round(float(theta_0), 6)  # точность до 6 знаков
        
        if cache_key in self._linearization_cache:
            return self._linearization_cache[cache_key]
        
        # Вычисляем матрицы только если их нет в кэше
        A_cont = np.array([
            [0.0, 1.0],
            [-self.g / self.l * np.cos(theta_0), -self.damping]
        ])
        
        B_cont = np.array([
            [0.0],
            [1.0]
        ])
        
        # Сохраняем в кэш
        result = (A_cont, B_cont)
        self._linearization_cache[cache_key] = result
        
        return result

    def discretize(self, A_cont: np.ndarray, B_cont: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Дискретизация непрерывной системы с помощью матричной экспоненты.
        """
        # Оптимизация: кэширование дорогой операции expm()
        A_hash = hash(A_cont.tobytes())
        B_hash = hash(B_cont.tobytes()) 
        dt_rounded = round(float(dt), 8)  # точность dt до 8 знаков
        cache_key = (A_hash, B_hash, dt_rounded)
        
        if cache_key in self._discretization_cache:
            return self._discretization_cache[cache_key]
        
        # Вычисляем только если нет в кэше
        n = A_cont.shape[0]
        m = B_cont.shape[1]
        
        augmented_matrix = np.zeros((n + m, n + m))
        augmented_matrix[0:n, 0:n] = A_cont
        augmented_matrix[0:n, n:n+m] = B_cont
        
        phi = expm(augmented_matrix * dt)  # Дорогая операция!
        
        A_discrete = phi[0:n, 0:n]
        B_discrete = phi[0:n, n:n+m]
        
        # Сохраняем в кэш
        result = (A_discrete, B_discrete)
        self._discretization_cache[cache_key] = result
        
        return result

    def discrete_step(self, state: np.ndarray, control: float, dt: float) -> np.ndarray:
        """
        Выполняет один шаг дискретной динамики.
        """
        A_cont, B_cont = self.get_linearized_matrices_at_state(state)
        A_discrete, B_discrete = self.discretize(A_cont, B_cont, dt)
        
        # Убедимся, что state и control имеют правильную форму
        state = np.asarray(state).reshape(-1, 1)
        control = np.asarray(control).reshape(-1, 1)

        next_state = A_discrete @ state + B_discrete @ control
        return next_state.flatten()
    
    def pendulum_dynamics(self, state: np.ndarray, control: float) -> np.ndarray:
        """
        Описывает непрерывную динамику нелинейного маятника.
        
        Args:
            state (np.ndarray): Текущее состояние [theta, theta_dot].
            control (float): Управляющее воздействие (крутящий момент).
            
        Returns:
            np.ndarray: Производная состояния [d_theta/dt, d_theta_dot/dt].
        """
        theta, theta_dot = state
        
        # Нелинейное уравнение движения маятника
        d_theta = theta_dot
        d_theta_dot = -self.g / self.l * np.sin(theta) - self.damping * theta_dot + control / (self.m * self.l**2)
        
        return np.array([d_theta, d_theta_dot])
    
    def third_derivative(self, state: np.ndarray, control: float, control_dot: float = 0.0) -> float:
        """
        Вычисляет третью производную угла маятника (ω̈).
        
        Args:
            state (np.ndarray): Текущее состояние [theta, theta_dot (omega)].
            control (float): Управляющее воздействие (крутящий момент).
            control_dot (float): Производная управляющего воздействия (по умолчанию 0).
            
        Returns:
            float: Третья производная угла θ (угловое ускорение ω̈).
        """
        theta, omega = state
        
        # Параметры системы
        g_over_l = self.g / self.l
        inv_ml2 = self._inv_ml2
        c = self.damping
        
        # Третья производная:
        # ω̈ = -(g/l)cos(θ)·ω + c·(g/l)sin(θ) + c²·ω - c·u/(m·l²) + u̇/(m·l²)
        third_deriv = (
            -g_over_l * np.cos(theta) * omega +     # -(g/l)cos(θ)·ω
            c * g_over_l * np.sin(theta) +          # c·(g/l)sin(θ)  
            c * c * omega -                         # c²·ω
            c * control * inv_ml2 +                 # -c·u/(m·l²)
            control_dot * inv_ml2                   # u̇/(m·l²)
        )
        
        return third_deriv
    
    def get_all_derivatives(self, state: np.ndarray, control: float, control_dot: float = 0.0) -> tuple[float, float, float]:
        """
        Возвращает все производные угла маятника до третьего порядка.
        
        Args:
            state (np.ndarray): Текущее состояние [theta, theta_dot].
            control (float): Управляющее воздействие.
            control_dot (float): Производная управляющего воздействия.
            
        Returns:
            Tuple[float, float, float]: (θ̇, θ̈, θ⃛) - первая, вторая и третья производные.
        """
        theta, omega = state
        
        # Первая производная
        theta_dot = omega
        
        # Вторая производная (из уравнений движения)
        theta_ddot = -self.g / self.l * np.sin(theta) - self.damping * omega + control / (self.m * self.l**2)
        
        # Третья производная
        theta_dddot = self.third_derivative(state, control, control_dot)
        
        return theta_dot, theta_ddot, theta_dddot


    
    def quad_step(self, state: np.ndarray, control: float, control_dot: float = 0.0, dt: float = 0.01) -> np.ndarray:
        theta_dot, theta_ddot, theta_dddot = self.get_all_derivatives(state, control, control_dot)
        linear_vector = np.array([theta_dot, theta_ddot])
        quad_vector = np.array([theta_dot,theta_dddot])
        
        next_state = state + linear_vector * dt + quad_vector * dt**2 / 2

        return next_state
        

    def scipy_rk45_step(self, state: np.ndarray, control: float, dt: float) -> np.ndarray:
        """
        Выполняет один шаг численного интегрирования с помощью solve_ivp (RK45).
        
        Args:
            state (np.ndarray): Текущее состояние [theta, theta_dot].
            control (float): Управляющее воздействие.
            dt (float): Размер временного шага.
            
        Returns:
            np.ndarray: Следующее состояние системы.
        """
        # solve_ivp решает систему от t_span[0] до t_span[1]
        # Мы хотим сделать всего один шаг, поэтому t_span = [0, dt]
        t_span = [0, dt]
        
        # y0 - начальное состояние
        y0 = state

        def dynamics_wrapper(t, y):
            return self.pendulum_dynamics(y, control)
        
        # Передаем функцию динамики и дополнительные аргументы (control)
        solution = solve_ivp(
            fun=dynamics_wrapper,
            t_span=t_span,
            y0=y0,
            method='RK45', 
            rtol=1e-6, # Относительный допуск по ошибке
            atol=1e-8 # Абсолютный допуск по ошибке
        )
        
        # Результат находится в последнем столбце массива y
        next_state = solution.y[:, -1]
        
        return next_state

    def find_all_quadratic_intersections(
        self,
        state1: np.ndarray,
        state2: np.ndarray,
        control1: float,
        control2: float,
        control_dot1: float = 0.0,
        control_dot2: float = 0.0,
        tolerance: float = 1e-6,
    ):
        """
        Находит все точки пересечения ("столкновения") квадратичных аппроксимаций 
        траекторий двух спор. Ищет моменты времени t1, t2, когда state1(t1) ≈ state2(t2).

        ИСПРАВЛЕНА ОШИБКА: теперь правильно учитывается, что траектории могут 
        пересекаться в разные моменты времени t1 и t2.

        Аргументы:
            state1, control1, control_dot1: Параметры первой споры.
            state2, control2, control_dot2: Параметры второй споры.
            tolerance: Допустимая погрешность для проверки решения.

        Возвращает:
            Список словарей с информацией о пересечениях:
            [{"t1": float, "t2": float, "point": np.array, "theta": float, "omega": float}, ...]
            t1, t2: Время до пересечения для каждой споры.
            point: Вектор состояния в точке пересечения [theta, omega].
            theta, omega: Координаты пересечения для удобства.
        """
        # Рассчитываем производные для обеих спор
        theta_dot1, theta_ddot1, theta_dddot1 = self.get_all_derivatives(state1, control1, control_dot1)
        theta_dot2, theta_ddot2, theta_dddot2 = self.get_all_derivatives(state2, control2, control_dot2)
        
        # Новый алгоритм: ищем пересечения траекторий в разные моменты времени

        # Аналитическое решение системы квадратных уравнений
        # 
        # Система уравнений:
        # θ₁₀ + θ̇₁₀·t₁ + 0.5·θ̈₁·t₁² = θ₂₀ + θ̇₂₀·t₂ + 0.5·θ̈₂·t₂²    (углы)
        # θ̇₁₀ + θ̈₁·t₁            = θ̇₂₀ + θ̈₂·t₂              (скорости)
        #
        # Из второго уравнения выражаем t₂:
        # t₂ = (θ̇₁₀ + θ̈₁·t₁ - θ̇₂₀) / θ̈₂
        #
        # Подставляем в первое уравнение и получаем квадратное уравнение на t₁
        
        all_intersections = []
        
        # Проверяем, не равно ли ускорение второй споры нулю
        if abs(theta_ddot2) < tolerance:
            # Случай θ̈₂ ≈ 0: вторая спора движется равномерно
            # Из второго уравнения: t₁ = (θ̇₂₀ - θ̇₁₀) / θ̈₁
            if abs(theta_ddot1) > tolerance:
                t1 = (state2[1] - state1[1]) / theta_ddot1
                t2 = state2[1]  # константная скорость
                
                # Проверяем первое уравнение (используем напрямую начальные состояния)
                theta1_t = state1[0] + state1[1] * t1 + 0.5 * theta_ddot1 * t1**2
                theta2_t = state2[0] + state2[1] * t2
                
                if abs(theta1_t - theta2_t) < tolerance:
                    intersection_point = np.array([(theta1_t + theta2_t)/2, state2[1]])
                    all_intersections.append({
                        "t1": t1,
                        "t2": t2, 
                        "point": intersection_point,
                        "theta": intersection_point[0],
                        "omega": intersection_point[1]
                    })
        else:
            # Общий случай: t₂ = (ω₁₀ + θ̈₁·t₁ - ω₂₀) / θ̈₂
            # Подставляем в уравнение для углов:
            # θ₁₀ + θ̇₁₀·t₁ + 0.5·θ̈₁·t₁² = θ₂₀ + θ̇₂₀·t₂ + 0.5·θ̈₂·t₂²
            #
            # Получаем квадратное уравнение на t₁: a·t₁² + b·t₁ + c = 0
            
            dtheta_0 = state1[0] - state2[0]     # разность начальных углов  
            domega_0 = state1[1] - state2[1]    # разность начальных скоростей
            
            # Точный вывод коэффициентов квадратного уравнения
            # После подстановки t₂ и упрощения получаем: a·t₁² + b·t₁ + c = 0
            
            # Обозначения для ясности
            w1, w2 = state1[1], state2[1]  # начальные скорости
            dd1, dd2 = theta_ddot1, theta_ddot2  # ускорения
            
            # ИСПРАВЛЕННЫЕ коэффициенты - полный пересчёт
            # Подставляем t₂ = (w1 + dd1*t₁ - w2)/dd2 в уравнение углов
            # После алгебраических упрощений получаем:
            a = 0.5 * (dd1 - dd1**2 / dd2)
            b = w1 * (1 - dd1 / dd2)  
            c = dtheta_0 - w2 * (w1 - w2) / dd2 - 0.5 * dd2 * ((w1 - w2) / dd2)**2
            
            # Решаем квадратное уравнение
            if abs(a) < tolerance:
                # Линейный случай
                if abs(b) > tolerance:
                    t1 = -c / b
                    t2 = (state1[1] + theta_ddot1 * t1 - state2[1]) / theta_ddot2
                    
                    # Проверяем решение (используем напрямую начальные состояния)
                    theta1_t = state1[0] + state1[1] * t1 + 0.5 * theta_ddot1 * t1**2
                    omega1_t = state1[1] + theta_ddot1 * t1
                    theta2_t = state2[0] + state2[1] * t2 + 0.5 * theta_ddot2 * t2**2
                    omega2_t = state2[1] + theta_ddot2 * t2
                    
                    if (abs(theta1_t - theta2_t) < tolerance and 
                        abs(omega1_t - omega2_t) < tolerance):
                        intersection_point = np.array([(theta1_t + theta2_t)/2, (omega1_t + omega2_t)/2])
                        all_intersections.append({
                            "t1": t1,
                            "t2": t2,
                            "point": intersection_point, 
                            "theta": intersection_point[0],
                            "omega": intersection_point[1]
                        })
            else:
                # Квадратный случай
                discriminant = b**2 - 4*a*c
                if discriminant >= 0:
                    sqrt_d = np.sqrt(discriminant)
                    
                    # Два решения
                    for t1 in [(-b + sqrt_d)/(2*a), (-b - sqrt_d)/(2*a)]:
                        t2 = (state1[1] + theta_ddot1 * t1 - state2[1]) / theta_ddot2
                        
                        # Проверяем решение (используем напрямую начальные состояния)
                        theta1_t = state1[0] + state1[1] * t1 + 0.5 * theta_ddot1 * t1**2
                        omega1_t = state1[1] + theta_ddot1 * t1
                        theta2_t = state2[0] + state2[1] * t2 + 0.5 * theta_ddot2 * t2**2
                        omega2_t = state2[1] + theta_ddot2 * t2
                        
                        if (abs(theta1_t - theta2_t) < tolerance and 
                            abs(omega1_t - omega2_t) < tolerance):
                            intersection_point = np.array([(theta1_t + theta2_t)/2, (omega1_t + omega2_t)/2])
                            all_intersections.append({
                                "t1": t1,
                                "t2": t2,
                                "point": intersection_point,
                                "theta": intersection_point[0], 
                                "omega": intersection_point[1]
                            })

        return all_intersections

    def scipy_rk45_step_backward(self, state: np.ndarray, control: float, dt: float) -> np.ndarray:
        """
        Простой шаг назад во времени - интегрируем с отрицательным dt.
        
        Args:
            state: Текущее состояние [theta, theta_dot]
            control: Управляющее воздействие  
            dt: Временной шаг (положительный)
            
        Returns:
            Состояние на dt секунд раньше
        """
        # Просто используем обычный метод с отрицательным временным шагом
        return self.scipy_rk45_step(state, control, -dt)
    

# ──────────────────────────────────────────────────────────────────────
    # 1. JIT-ядро: одиночный RK4–шаг (fastmath + параллель разрешён)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    @njit(float64[:](float64[:], float64, float64,          # state, u, dt
                     float64, float64, float64, float64),   # g, l, c, inv_ml2
          cache=True, fastmath=True)
    def _rk4_step(state, u, dt, g, l, c, inv_ml2):
        th, om = state
        k1t, k1o = om, -g / l * np.sin(th) - c * om + u * inv_ml2
        k2t, k2o = om + 0.5 * dt * k1o, -g / l * np.sin(th + 0.5 * dt * k1t) - c * (om + 0.5 * dt * k1o) + u * inv_ml2
        k3t, k3o = om + 0.5 * dt * k2o, -g / l * np.sin(th + 0.5 * dt * k2t) - c * (om + 0.5 * dt * k2o) + u * inv_ml2
        k4t, k4o = om + dt * k3o,       -g / l * np.sin(th + dt * k3t)       - c * (om + dt * k3o)       + u * inv_ml2
        th_n = th + (dt / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)
        om_n = om + (dt / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o)
        return np.array([th_n, om_n])

    # ──────────────────────────────────────────────────────────────────────
    # 2. ПАКЕТНЫЙ шаг для векторных вычислений (параллельный prange)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
    def _batch_rk4(states, controls, dts, g, l, c, inv_ml2):
        out = np.empty_like(states)
        for i in prange(states.shape[0]):
            th, om = states[i]
            u, dt = controls[i], dts[i]

            k1t, k1o = om, -g / l * np.sin(th) - c * om + u * inv_ml2
            k2t, k2o = om + 0.5 * dt * k1o, -g / l * np.sin(th + 0.5 * dt * k1t) - c * (om + 0.5 * dt * k1o) + u * inv_ml2
            k3t, k3o = om + 0.5 * dt * k2o, -g / l * np.sin(th + 0.5 * dt * k2t) - c * (om + 0.5 * dt * k2o) + u * inv_ml2
            k4t, k4o = om + dt * k3o,       -g / l * np.sin(th + dt * k3t)       - c * (om + dt * k3o)       + u * inv_ml2

            out[i, 0] = th + (dt / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)
            out[i, 1] = om + (dt / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o)
        return out

    # ──────────────────────────────────────────────────────────────────────
    # 3. Публичный одиночный шаг
    # ──────────────────────────────────────────────────────────────────────
    def step(self, state: np.ndarray, control: float, dt: float, method: str = "jit") -> np.ndarray:
        """
        Выполняет один интеграционный шаг.
        method = "jit"  (быстро)  или  "rk45" (fallback SciPy, медленно).
        """
        if method == "jit":
            return self._rk4_step(state, control, dt, self.g, self.l, self.damping, self._inv_ml2)
        elif method == "rk45":
            from scipy.integrate import RK45

            def f(_, y):
                th, om = y
                dtheta = om
                domega = -self.g / self.l * np.sin(th) - self.damping * om + control * self._inv_ml2
                return np.array([dtheta, domega])

            solver = RK45(f, 0.0, state, dt, max_step=dt)
            solver.step()
            return solver.y
        else:
            raise ValueError("method must be 'jit' or 'rk45'")

    # ──────────────────────────────────────────────────────────────────────
    # 4. Публичный batch-шаг (используйте его в SporeTree)
    # ──────────────────────────────────────────────────────────────────────
    def batch_step(self, states: np.ndarray, controls: np.ndarray, dts: np.ndarray) -> np.ndarray:
        """
        Параллельный расчёт множества траекторий за один вызов.
        states   : (N, 2)
        controls : (N,)
        dts      : (N,)
        """
        return self._batch_rk4(states, controls, dts, self.g, self.l, self.damping, self._inv_ml2)

    # ──────────────────────────────────────────────────────────────────────
