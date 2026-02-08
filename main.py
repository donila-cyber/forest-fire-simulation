import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime
import json
import time
import threading
from enum import Enum


class FireStatus(Enum):
    """Статусы пожара"""
    NORMAL = "Нормальный"
    SPREADING = "Распространяется"
    CRITICAL = "Критический"
    CONTROLLED = "Контролируемый"
    EXTINGUISHED = "Потухший"


class EnhancedForestFireModel:
    """Улучшенная модель лесного пожара с расширенным функционалом"""

    def __init__(self, grid_size=100, dt=0.1, dx=1.0):
        # Основные параметры модели
        self.grid_size = grid_size
        self.dt = dt
        self.dx = dx

        # Физические параметры (теперь с диапазонами)
        self.params = {
            'D_T': {'value': 0.2, 'min': 0.01, 'max': 1.0, 'desc': 'Коэффициент температуропроводности'},
            'A': {'value': 8.0, 'min': 1.0, 'max': 20.0, 'desc': 'Теплота сгорания'},
            'B': {'value': 0.1, 'min': 0.01, 'max': 0.5, 'desc': 'Коэффициент теплопотерь'},
            'C': {'value': 0.5, 'min': 0.1, 'max': 2.0, 'desc': 'Скорость выгорания топлива'},
            'T_c': {'value': 1.0, 'min': 0.5, 'max': 3.0, 'desc': 'Критическая температура'},
            'T_0': {'value': 0.0, 'min': -5.0, 'max': 5.0, 'desc': 'Температура окружающей среды'},
            'F_0': {'value': 1.0, 'min': 0.1, 'max': 3.0, 'desc': 'Начальная плотность топлива'},
        }

        # Параметры ветра (теперь может меняться со временем)
        self.wind = {
            'x': 0.5,
            'y': 0.2,
            'variability': 0.1,  # Изменчивость ветра
            'last_change': 0,
            'change_interval': 50  # Шаги между изменениями ветра
        }

        # Инициализация полей
        self.reset_fields()

        # История моделирования
        self.history = {
            'time': [],
            'burned_area': [],
            'max_temperature': [],
            'active_fires': [],
            'fuel_remaining': []
        }

        # Статистика
        self.stats = {
            'total_burned': 0,
            'max_temperature_reached': 0,
            'simulation_time': 0,
            'steps_completed': 0,
            'fire_status': FireStatus.NORMAL
        }

        # Система событий
        self.events = []

        # Флаги состояния
        self.is_running = False
        self.is_paused = False
        self.simulation_speed = 1.0

    def reset_fields(self):
        """Сброс всех полей модели"""
        # Сначала инициализируем базовые атрибуты
        self.time = 0.0
        self.step_count = 0

        # Инициализация полей
        self.T = np.zeros((self.grid_size, self.grid_size))  # Температура
        self.F = np.ones((self.grid_size, self.grid_size)) * self.params['F_0']['value']  # Топливо
        self.burned = np.zeros((self.grid_size, self.grid_size), dtype=bool)  # Выгоревшие области
        self.wind_field = np.zeros((self.grid_size, self.grid_size, 2))  # Векторное поле ветра

        # Инициализация случайного ветра
        self.update_wind_field()

    def update_wind_field(self):
        """Обновление векторного поля ветра"""
        # Основное направление
        base_wind = np.array([self.wind['x'], self.wind['y']])

        # Добавляем случайные флуктуации
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                noise = np.random.randn(2) * self.wind['variability']
                self.wind_field[i, j] = base_wind + noise

        # Периодическое изменение направления ветра
        # Убедимся, что step_count существует
        if hasattr(self, 'step_count') and self.step_count - self.wind['last_change'] > self.wind['change_interval']:
            self.wind['x'] += np.random.uniform(-0.2, 0.2)
            self.wind['y'] += np.random.uniform(-0.2, 0.2)
            self.wind['last_change'] = self.step_count

    def initialize_fire(self, center_x=None, center_y=None, radius=3, intensity=2.0):
        """Инициализация очага пожара"""
        if center_x is None:
            center_x = self.grid_size // 2
        if center_y is None:
            center_y = self.grid_size // 2

        # Создаем круглый очаг пожара
        for i in range(max(0, center_x - radius), min(self.grid_size, center_x + radius + 1)):
            for j in range(max(0, center_y - radius), min(self.grid_size, center_y + radius + 1)):
                distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                if distance <= radius:
                    self.T[i, j] = intensity

    def initialize_random_fire(self, num_fires=3, max_radius=5):
        """Случайная инициализация нескольких очагов"""
        for _ in range(num_fires):
            x = np.random.randint(max_radius, self.grid_size - max_radius)
            y = np.random.randint(max_radius, self.grid_size - max_radius)
            radius = np.random.randint(1, max_radius)
            intensity = np.random.uniform(1.5, 3.0)
            self.initialize_fire(x, y, radius, intensity)

    def initialize_terrain(self, terrain_type='random', **kwargs):
        """Инициализация рельефа местности"""
        if terrain_type == 'random':
            variability = kwargs.get('variability', 0.3)
            noise = 1.0 + variability * (np.random.random((self.grid_size, self.grid_size)) - 0.5)
            self.F *= noise

        elif terrain_type == 'gradient':
            # Градиент плотности (например, от реки к лесу)
            x = np.linspace(0, 1, self.grid_size)
            y = np.linspace(0, 1, self.grid_size)
            X, Y = np.meshgrid(x, y)
            gradient = 0.5 + 0.5 * (X + Y)
            self.F *= gradient

        elif terrain_type == 'from_file':
            filename = kwargs.get('filename')
            if filename:
                try:
                    data = np.loadtxt(filename)
                    if data.shape == (self.grid_size, self.grid_size):
                        self.F = data
                except:
                    print(f"Ошибка загрузки файла {filename}")

        self.F = np.clip(self.F, 0.1, 3.0)

    def upwind_derivative(self, field, wind_component, axis):
        """Upwind-разность для адвективного члена"""
        if axis == 0:  # x-направление
            if wind_component >= 0:
                return (field - np.roll(field, 1, axis=0)) / self.dx
            else:
                return (np.roll(field, -1, axis=0) - field) / self.dx
        else:  # y-направление
            if wind_component >= 0:
                return (field - np.roll(field, 1, axis=1)) / self.dx
            else:
                return (np.roll(field, -1, axis=1) - field) / self.dx

    def vector_advection(self, field):
        """Адвекция с векторным полем ветра"""
        adv_x = -self.wind_field[:, :, 0] * self.upwind_derivative(field, 1, 0)
        adv_y = -self.wind_field[:, :, 1] * self.upwind_derivative(field, 1, 1)
        return adv_x + adv_y

    def laplacian(self, field):
        """Оператор Лапласа (диффузия)"""
        return (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
                np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
                4 * field) / (self.dx ** 2)

    def reaction_function(self, T):
        """Функция скорости реакции (пороговая)"""
        return (T >= self.params['T_c']['value']).astype(float)

    def apply_boundary_conditions(self, field):
        """Граничные условия Неймана (нулевой поток через границу)"""
        field[0, :] = field[1, :]
        field[-1, :] = field[-2, :]
        field[:, 0] = field[:, 1]
        field[:, -1] = field[:, -2]
        return field

    def step(self):
        """Один шаг по времени"""
        if not self.is_running or self.is_paused:
            return

        start_time = time.time()

        # Обновляем ветер
        self.update_wind_field()

        # Создаем копии для вычислений
        T_new = self.T.copy()
        F_new = self.F.copy()

        # Вычисляем члены уравнения для температуры
        diffusion = self.params['D_T']['value'] * self.laplacian(self.T)
        advection = self.vector_advection(self.T)

        Y = self.reaction_function(self.T)
        reaction = self.params['A']['value'] * self.F * Y
        loss = self.params['B']['value'] * (self.T - self.params['T_0']['value'])

        # Обновляем температуру
        T_new += self.dt * (diffusion + advection + reaction - loss)

        # Обновляем плотность топлива
        F_new -= self.dt * self.params['C']['value'] * self.F * Y
        F_new = np.clip(F_new, 0, None)

        # Обновляем маску выгоревших областей
        new_burned = F_new < 0.01
        self.burned = np.logical_or(self.burned, new_burned)

        # Применяем граничные условия
        T_new = self.apply_boundary_conditions(T_new)
        F_new = self.apply_boundary_conditions(F_new)

        self.T = T_new
        self.F = F_new
        self.time += self.dt
        self.step_count += 1

        # Обновляем статистику
        self.update_statistics()

        # Проверяем события
        self.check_events()

        calc_time = time.time() - start_time
        time.sleep(max(0, (self.dt / self.simulation_speed) - calc_time))

    def update_statistics(self):
        """Обновление статистики модели"""
        burned_area = np.sum(self.burned) / (self.grid_size ** 2)
        max_temp = self.T.max()
        active_fires = np.sum(self.T > self.params['T_c']['value'])
        fuel_remaining = np.mean(self.F)

        # Сохраняем историю
        self.history['time'].append(self.time)
        self.history['burned_area'].append(burned_area)
        self.history['max_temperature'].append(max_temp)
        self.history['active_fires'].append(active_fires)
        self.history['fuel_remaining'].append(fuel_remaining)

        # Обновляем статистику
        self.stats['total_burned'] = burned_area
        self.stats['max_temperature_reached'] = max(self.stats['max_temperature_reached'], max_temp)
        self.stats['simulation_time'] = self.time
        self.stats['steps_completed'] = self.step_count

        # Определяем статус пожара
        if burned_area < 0.1:
            self.stats['fire_status'] = FireStatus.NORMAL
        elif burned_area < 0.3:
            self.stats['fire_status'] = FireStatus.SPREADING
        elif burned_area < 0.6:
            self.stats['fire_status'] = FireStatus.CRITICAL
        elif active_fires == 0:
            self.stats['fire_status'] = FireStatus.EXTINGUISHED
        else:
            self.stats['fire_status'] = FireStatus.CONTROLLED

    def check_events(self):
        """Проверка событий (критические температуры, скорость распространения)"""
        burned_area = self.history['burned_area'][-1]

        # Критическая температура
        if self.history['max_temperature'][-1] > 3.0:
            self.add_event(f"Критическая температура: {self.history['max_temperature'][-1]:.2f}")

        # Быстрое распространение
        if len(self.history['burned_area']) > 10:
            recent_spread = self.history['burned_area'][-1] - self.history['burned_area'][-10]
            if recent_spread > 0.05:
                self.add_event(f"Быстрое распространение: {recent_spread * 100:.1f}% за 10 шагов")

        # Критическая площадь выгорания
        if burned_area > 0.5:
            self.add_event(f"Выгорело более 50% площади: {burned_area * 100:.1f}%")

    def add_event(self, message):
        """Добавление события в лог"""
        event = {
            'time': self.time,
            'message': message,
            'step': self.step_count
        }
        self.events.append(event)
        print(f"[{event['time']:.1f}с] {message}")

    def get_fire_intensity(self):
        """Интенсивность пожара для визуализации"""
        intensity = self.T.copy()
        intensity[self.burned] = -0.5  # Выгоревшие области
        return intensity

    def get_wind_vectors(self, step=5):
        """Получить векторы ветра для отображения"""
        x = np.arange(0, self.grid_size, step)
        y = np.arange(0, self.grid_size, step)
        X, Y = np.meshgrid(x, y)
        U = self.wind_field[::step, ::step, 0]
        V = self.wind_field[::step, ::step, 1]
        return X, Y, U, V

    def save_state(self, filename):
        """Сохранение состояния модели"""
        state = {
            'grid_size': self.grid_size,
            'dt': self.dt,
            'dx': self.dx,
            'params': self.params,
            'wind': self.wind,
            'T': self.T.tolist(),
            'F': self.F.tolist(),
            'burned': self.burned.tolist(),
            'time': self.time,
            'step_count': self.step_count,
            'history': self.history,
            'stats': {
                'total_burned': self.stats['total_burned'],
                'max_temperature_reached': self.stats['max_temperature_reached'],
                'simulation_time': self.stats['simulation_time'],
                'steps_completed': self.stats['steps_completed'],
                'fire_status': self.stats['fire_status'].value
            },
            'events': self.events
        }

        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filename):
        """Загрузка состояния модели"""
        with open(filename, 'r') as f:
            state = json.load(f)

        self.grid_size = state['grid_size']
        self.dt = state['dt']
        self.dx = state['dx']
        self.params = state['params']
        self.wind = state['wind']
        self.T = np.array(state['T'])
        self.F = np.array(state['F'])
        self.burned = np.array(state['burned'])
        self.time = state['time']
        self.step_count = state['step_count']
        self.history = state['history']
        self.stats = state['stats']
        self.stats['fire_status'] = FireStatus(state['stats']['fire_status'])
        self.events = state['events']


class ForestFireGUI:
    """Графический интерфейс для модели лесного пожара"""

    def __init__(self):
        self.model = EnhancedForestFireModel(grid_size=100)
        self.create_gui()

    def create_gui(self):
        """Создание графического интерфейса"""
        self.root = tk.Tk()
        self.root.title("Моделирование лесного пожара")
        self.root.geometry("1400x900")

        # Создаем стиль
        style = ttk.Style()
        style.theme_use('clam')

        # Главный контейнер
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая панель: управление
        left_panel = ttk.LabelFrame(main_frame, text="Управление", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Параметры модели
        params_frame = ttk.LabelFrame(left_panel, text="Параметры модели", padding=5)
        params_frame.pack(fill=tk.X, pady=(0, 10))

        self.param_vars = {}
        row = 0

        for param_name, param_data in self.model.params.items():
            ttk.Label(params_frame, text=f"{param_data['desc']}:").grid(row=row, column=0, sticky=tk.W, pady=2)

            var = tk.DoubleVar(value=param_data['value'])
            self.param_vars[param_name] = var

            scale = ttk.Scale(
                params_frame,
                from_=param_data['min'],
                to=param_data['max'],
                variable=var,
                orient=tk.HORIZONTAL,
                length=200
            )
            scale.grid(row=row, column=1, padx=5, pady=2)

            value_label = ttk.Label(params_frame, text=f"{param_data['value']:.2f}")
            value_label.grid(row=row, column=2, padx=5)

            # Привязка изменения значения
            var.trace('w', lambda name, index, mode, p=param_name, l=value_label:
            self.update_param_display(p, l))

            row += 1

        # Параметры ветра
        wind_frame = ttk.LabelFrame(left_panel, text="Ветер", padding=5)
        wind_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(wind_frame, text="X:").grid(row=0, column=0, sticky=tk.W)
        self.wind_x_var = tk.DoubleVar(value=self.model.wind['x'])
        ttk.Scale(wind_frame, from_=-2, to=2, variable=self.wind_x_var,
                  orient=tk.HORIZONTAL, length=150).grid(row=0, column=1)
        ttk.Label(wind_frame, textvariable=tk.StringVar(value=f"{self.model.wind['x']:.2f}")).grid(row=0, column=2)

        ttk.Label(wind_frame, text="Y:").grid(row=1, column=0, sticky=tk.W)
        self.wind_y_var = tk.DoubleVar(value=self.model.wind['y'])
        ttk.Scale(wind_frame, from_=-2, to=2, variable=self.wind_y_var,
                  orient=tk.HORIZONTAL, length=150).grid(row=1, column=1)
        ttk.Label(wind_frame, textvariable=tk.StringVar(value=f"{self.model.wind['y']:.2f}")).grid(row=1, column=2)

        # Кнопки управления
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_btn = ttk.Button(control_frame, text="Старт", command=self.start_simulation)
        self.start_btn.pack(side=tk.LEFT, padx=2)

        self.pause_btn = ttk.Button(control_frame, text="Пауза", command=self.pause_simulation, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=2)

        self.stop_btn = ttk.Button(control_frame, text="Стоп", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        self.reset_btn = ttk.Button(control_frame, text="Сброс", command=self.reset_simulation)
        self.reset_btn.pack(side=tk.LEFT, padx=2)

        # Скорость симуляции
        speed_frame = ttk.LabelFrame(left_panel, text="Скорость симуляции", padding=5)
        speed_frame.pack(fill=tk.X, pady=(0, 10))

        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Scale(speed_frame, from_=0.1, to=5.0, variable=self.speed_var,
                  orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(speed_frame, textvariable=tk.StringVar(value="1.0x")).pack()

        # Инициализация пожара
        fire_frame = ttk.LabelFrame(left_panel, text="Инициализация пожара", padding=5)
        fire_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(fire_frame, text="Случайные очаги",
                   command=lambda: self.init_fire('random')).pack(fill=tk.X, pady=2)
        ttk.Button(fire_frame, text="Центральный очаг",
                   command=lambda: self.init_fire('center')).pack(fill=tk.X, pady=2)
        ttk.Button(fire_frame, text="Несколько очагов",
                   command=lambda: self.init_fire('multiple')).pack(fill=tk.X, pady=2)

        # Тип местности
        terrain_frame = ttk.LabelFrame(left_panel, text="Тип местности", padding=5)
        terrain_frame.pack(fill=tk.X, pady=(0, 10))

        self.terrain_var = tk.StringVar(value="random")
        terrains = [("Случайная", "random"), ("Градиент", "gradient"), ("Равномерная", "uniform")]

        for text, value in terrains:
            ttk.Radiobutton(terrain_frame, text=text, value=value,
                            variable=self.terrain_var).pack(anchor=tk.W)

        # Сохранение/загрузка
        io_frame = ttk.LabelFrame(left_panel, text="Сохранение/Загрузка", padding=5)
        io_frame.pack(fill=tk.X)

        ttk.Button(io_frame, text="Сохранить состояние",
                   command=self.save_state).pack(fill=tk.X, pady=2)
        ttk.Button(io_frame, text="Загрузить состояние",
                   command=self.load_state).pack(fill=tk.X, pady=2)
        ttk.Button(io_frame, text="Экспорт данных",
                   command=self.export_data).pack(fill=tk.X, pady=2)

        # Правая панель: визуализация и информация
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Вкладки
        notebook = ttk.Notebook(right_panel)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка визуализации
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="Визуализация")

        # Создаем фигуру matplotlib
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Настройка цветовых схем
        self.cmap_temp = colors.ListedColormap(['black', 'darkgreen', 'yellow', 'orange', 'red', 'white'])
        bounds_temp = [-1, 0, 0.5, 1.0, 2.0, 3.0, 10]
        self.norm_temp = colors.BoundaryNorm(bounds_temp, self.cmap_temp.N)

        self.cmap_fuel = plt.cm.Greens
        self.cmap_fuel.set_under('black')

        self.cmap_fire = colors.ListedColormap(['black', 'darkgreen', 'yellow', 'orange', 'red', 'white'])
        bounds_fire = [-1, -0.1, 0, 0.5, 1.0, 2.0, 10]
        self.norm_fire = colors.BoundaryNorm(bounds_fire, self.cmap_fire.N)

        # Первоначальный рендеринг
        self.im1 = self.ax1.imshow(self.model.T, cmap=self.cmap_temp, norm=self.norm_temp)
        self.im2 = self.ax2.imshow(self.model.F, cmap=self.cmap_fuel, vmin=0, vmax=2)
        self.im3 = self.ax3.imshow(self.model.get_fire_intensity(), cmap=self.cmap_fire, norm=self.norm_fire)

        # Векторы ветра
        X, Y, U, V = self.model.get_wind_vectors(step=10)
        self.quiver = self.ax4.quiver(X, Y, U, V, color='white', scale=30)

        # Настройка графиков
        titles = ['Температура', 'Плотность топлива', 'Интенсивность пожара', 'Векторы ветра']
        for ax, title in zip([self.ax1, self.ax2, self.ax3, self.ax4], titles):
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        self.fig.tight_layout()

        # Вкладка "Статистика" 
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Статистика")

        # Текстовое поле
        stats_text = scrolledtext.ScrolledText(stats_frame, width=60, height=22, font=("Consolas", 10))
        stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        # Кнопка сохранения
        btn_save_stats = ttk.Button(
            stats_frame,
            text="💾 Сохранить статистику",
            command=self.save_stats_to_file
        )
        btn_save_stats.pack(pady=(0, 10))

        self.stats_text = stats_text

        # Вкладка "События"
        events_frame = ttk.Frame(notebook)
        notebook.add(events_frame, text="События")

        # Текстовое поле
        events_text = scrolledtext.ScrolledText(events_frame, width=60, height=22, font=("Consolas", 10))
        events_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        # Кнопка сохранения
        btn_save_events = ttk.Button(
            events_frame,
            text="📥 Сохранить события",
            command=self.save_events_to_file
        )
        btn_save_events.pack(pady=(0, 10))

        self.events_text = events_text

        # Вкладка графиков
        plots_frame = ttk.Frame(notebook)
        notebook.add(plots_frame, text="Графики")

        self.fig2, ((self.ax5, self.ax6), (self.ax7, self.ax8)) = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=plots_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Информационная панель внизу
        info_frame = ttk.LabelFrame(right_panel, text="Информация", padding=5)
        info_frame.pack(fill=tk.X, pady=(10, 0))

        self.info_var = tk.StringVar()
        info_label = ttk.Label(info_frame, textvariable=self.info_var, font=('Arial', 10))
        info_label.pack()

        # Запуск обновления интерфейса
        self.update_interval = 100  # мс
        self.root.after(self.update_interval, self.update_gui)

    def update_param_display(self, param_name, label):
        """Обновление отображения параметра"""
        value = self.param_vars[param_name].get()
        label.config(text=f"{value:.2f}")
        self.model.params[param_name]['value'] = value

    def init_fire(self, fire_type):
        """Инициализация пожара"""
        self.model.reset_fields()

        # Инициализация местности
        terrain_type = self.terrain_var.get()
        self.model.initialize_terrain(terrain_type)

        # Инициализация пожара
        if fire_type == 'random':
            self.model.initialize_random_fire()
        elif fire_type == 'center':
            self.model.initialize_fire()
        elif fire_type == 'multiple':
            self.model.initialize_random_fire(num_fires=5)

        self.update_visualization()

    def start_simulation(self):
        """Запуск симуляции"""
        if not self.model.is_running:
            self.model.is_running = True
            self.model.is_paused = False

            # Обновление параметров из GUI
            for param_name, var in self.param_vars.items():
                self.model.params[param_name]['value'] = var.get()

            self.model.wind['x'] = self.wind_x_var.get()
            self.model.wind['y'] = self.wind_y_var.get()

            # Запуск симуляции в отдельном потоке
            self.simulation_thread = threading.Thread(target=self.run_simulation, daemon=True)
            self.simulation_thread.start()

            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)

    def pause_simulation(self):
        """Пауза симуляции"""
        self.model.is_paused = not self.model.is_paused
        if self.model.is_paused:
            self.pause_btn.config(text="Продолжить")
        else:
            self.pause_btn.config(text="Пауза")

    def stop_simulation(self):
        """Остановка симуляции"""
        self.model.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="Пауза")
        self.stop_btn.config(state=tk.DISABLED)

    def reset_simulation(self):
        """Сброс симуляции"""
        self.stop_simulation()
        self.model.reset_fields()
        self.model.initialize_terrain(self.terrain_var.get())
        self.update_visualization()

    def run_simulation(self):
        """Основной цикл симуляции"""
        while self.model.is_running:
            self.model.step()
            self.model.simulation_speed = self.speed_var.get()

    def update_gui(self):
        """Обновление графического интерфейса"""
        if self.model.is_running:
            self.update_visualization()
            self.update_stats()
            self.update_events()
            self.update_plots()

        self.root.after(self.update_interval, self.update_gui)

    def update_visualization(self):
        """Обновление визуализации"""
        self.im1.set_array(self.model.T)
        self.im2.set_array(self.model.F)
        self.im3.set_array(self.model.get_fire_intensity())

        # Обновление векторов ветра
        X, Y, U, V = self.model.get_wind_vectors(step=10)
        self.quiver.set_UVC(U, V)

        self.canvas.draw()

    def update_stats(self):
        """Обновление статистики"""
        stats = self.model.stats
        status_icon = {
            FireStatus.NORMAL: "🟢",
            FireStatus.SPREADING: "🟡",
            FireStatus.CRITICAL: "🔥",
            FireStatus.CONTROLLED: "🟠",
            FireStatus.EXTINGUISHED: "💧"
        }.get(stats['fire_status'], "❓")

        text = f"""📊 СТАТИСТИКА МОДЕЛИРОВАНИЯ

    {status_icon} Статус пожара: {stats['fire_status'].value}
    ⏱️ Время симуляции: {stats['simulation_time']:.1f} с
    🔢 Выполнено шагов: {stats['steps_completed']}
    🌡️ Макс. температура: {stats['max_temperature_reached']:.2f}
    🌲 Выгоревшая площадь: {stats['total_burned'] * 100:.1f}%
    🔥 Текущая темп.: {self.model.T.max():.2f}
    📍 Активных очагов: {np.sum(self.model.T > self.model.params['T_c']['value'])}
    ⛽ Остаток топлива: {np.mean(self.model.F) * 100:.1f}%
    💨 Ветер: X={self.model.wind['x']:.2f}, Y={self.model.wind['y']:.2f}
    🌀 Скорость ветра: {np.sqrt(self.model.wind['x'] ** 2 + self.model.wind['y'] ** 2):.2f}
    """
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, text)

        # Информационная строка (внизу окна)
        info = f"⏱️ {stats['simulation_time']:.1f}с | 🔥 {self.model.T.max():.2f} | 🌲 {stats['total_burned'] * 100:.1f}% | {status_icon} {stats['fire_status'].value}"
        self.info_var.set(info)

    def update_events(self):
        """Обновление событий"""
        self.events_text.delete(1.0, tk.END)

        if not self.model.events:
            self.events_text.insert(tk.END, "📭 Нет событий\n")
            return

        self.events_text.insert(tk.END, "🔔 ЖУРНАЛ СОБЫТИЙ (последние 20):\n\n")

        for event in self.model.events[-20:]:
            icon = "🔥" if "температура" in event['message'].lower() else \
                "💨" if "ветер" in event['message'].lower() else \
                "⚠️"
            line = f"{icon} [{event['time']:.1f}с] {event['message']}\n"
            self.events_text.insert(tk.END, line)

    def update_plots(self):
        """Обновление графиков"""
        history = self.model.history

        if len(history['time']) > 1:
            # Очистка предыдущих графиков
            for ax in [self.ax5, self.ax6, self.ax7, self.ax8]:
                ax.clear()

            # График выгоревшей площади
            self.ax5.plot(history['time'], np.array(history['burned_area']) * 100)
            self.ax5.set_title('Выгоревшая площадь')
            self.ax5.set_xlabel('Время, с')
            self.ax5.set_ylabel('Площадь, %')
            self.ax5.grid(True, alpha=0.3)

            # График максимальной температуры
            self.ax6.plot(history['time'], history['max_temperature'])
            self.ax6.set_title('Максимальная температура')
            self.ax6.set_xlabel('Время, с')
            self.ax6.set_ylabel('Температура')
            self.ax6.grid(True, alpha=0.3)

            # График активных очагов
            self.ax7.plot(history['time'], history['active_fires'])
            self.ax7.set_title('Активные очаги')
            self.ax7.set_xlabel('Время, с')
            self.ax7.set_ylabel('Количество')
            self.ax7.grid(True, alpha=0.3)

            # График остатка топлива
            self.ax8.plot(history['time'], np.array(history['fuel_remaining']) * 100)
            self.ax8.set_title('Остаток топлива')
            self.ax8.set_xlabel('Время, с')
            self.ax8.set_ylabel('Топливо, %')
            self.ax8.grid(True, alpha=0.3)

            self.fig2.tight_layout()
            self.canvas2.draw()

    def save_state(self):
        """Сохранение состояния модели"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.model.save_state(filename)
            messagebox.showinfo("Сохранение", f"Состояние сохранено в {filename}")

    def load_state(self):
        """Загрузка состояния модели"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.model.load_state(filename)

            # Обновление интерфейса
            for param_name, var in self.param_vars.items():
                var.set(self.model.params[param_name]['value'])

            self.wind_x_var.set(self.model.wind['x'])
            self.wind_y_var.set(self.model.wind['y'])

            self.update_visualization()
            messagebox.showinfo("Загрузка", f"Состояние загружено из {filename}")

    def export_data(self):
        """Экспорт данных"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            # Экспорт истории
            import pandas as pd
            df = pd.DataFrame(self.model.history)
            df.to_csv(filename, index=False)

            # Экспорт текущего состояния
            state_filename = filename.replace('.csv', '_state.csv')
            with open(state_filename, 'w') as f:
                f.write("Параметр,Значение\n")
                for param_name, param_data in self.model.params.items():
                    f.write(f"{param_data['desc']},{param_data['value']}\n")
                f.write(f"Ветер X,{self.model.wind['x']}\n")
                f.write(f"Ветер Y,{self.model.wind['y']}\n")
                f.write(f"Время симуляции,{self.model.time}\n")
                f.write(f"Выгоревшая площадь,{self.model.stats['total_burned']}\n")

            messagebox.showinfo("Экспорт", f"Данные экспортированы в {filename}")

    def save_stats_to_file(self):
        """Сохранить содержимое вкладки 'Статистика' в файл"""
        content = self.stats_text.get(1.0, tk.END).strip()
        if not content or "Нет данных" in content:
            messagebox.showwarning("Сохранение", "Нет данных для сохранения.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")],
            initialfile="статистика_пожара.txt"
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            messagebox.showinfo("✅ Успешно", f"Статистика сохранена в:\n{filename}")

    def save_events_to_file(self):
        """Сохранить содержимое вкладки 'События' в файл"""
        content = self.events_text.get(1.0, tk.END).strip()
        if not content or "Нет событий" in content:
            messagebox.showwarning("Сохранение", "Нет событий для сохранения.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")],
            initialfile="события_пожара.txt"
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            messagebox.showinfo("✅ Успешно", f"События сохранены в:\n{filename}")
    def run(self):
        """Запуск главного цикла"""
        self.root.mainloop()


def main():
    """Основная функция"""
    print("Запуск улучшенной модели лесного пожара с GUI...")

    # Создаем и запускаем GUI
    gui = ForestFireGUI()
    gui.run()


if __name__ == "__main__":
    # Добавим недостающий импорт
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    main()