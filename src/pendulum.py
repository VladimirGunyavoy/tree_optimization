import numpy as np
from scipy.linalg import expm
from typing import Tuple
from scipy.integrate import solve_ivp

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