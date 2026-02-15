def simulate_fire_spread(initial_fire_points, terrain_data, wind_direction, wind_speed, time_steps):
    """
    Базовая функция для имитации распространения лесного пожара.
    На данном этапе возвращает упрощенные данные.

    Параметры:
    initial_fire_points (list): Список начальных точек возгорания.
    terrain_data (dict): Данные о местности (например, тип растительности, высота).
    wind_direction (str): Направление ветра (например, 'N', 'NE', 'E').
    wind_speed (float): Скорость ветра в м/с.
    time_steps (int): Количество временных шагов для симуляции.

    Возвращает:
    list: Упрощенные данные о распространении огня (например, список затронутых координат).
    """
    print(f"Симуляция пожара с {len(initial_fire_points)} начальными точками, ветром {wind_direction} {wind_speed} м/с.")
    # Упрощенная логика симуляции
    affected_area = []
    for point in initial_fire_points:
        # Простое расширение области вокруг каждой точки
        for step in range(time_steps):
            # Имитация распространения
            new_x = point[0] + step * 0.1 * wind_speed
            new_y = point[1] + step * 0.1 * wind_speed
            affected_area.append((new_x, new_y))
    return affected_area

if __name__ == '__main__':
    # Пример использования
    initial_points = [(0, 0), (10, 10)]
    terrain = {'type': 'forest', 'density': 0.8}
    wind_dir = 'E'
    wind_spd = 5.0
    steps = 10

    result = simulate_fire_spread(initial_points, terrain, wind_dir, wind_spd, steps)
    print(f"Результат симуляции: {result[:5]}...") # Выводим первые 5 элементов для краткости
