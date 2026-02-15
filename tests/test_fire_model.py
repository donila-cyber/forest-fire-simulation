import unittest
from src.fire_model import simulate_fire_spread

class TestFireModel(unittest.TestCase):

    def test_simulate_fire_spread_basic(self):
        initial_points = [(0, 0)]
        terrain = {"type": "forest", "density": 0.8}
        wind_dir = "N"
        wind_spd = 1.0
        time_steps = 1
        
        result = simulate_fire_spread(initial_points, terrain, wind_dir, wind_spd, time_steps)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIsInstance(result[0], tuple)
        self.assertEqual(len(result[0]), 2)

    def test_simulate_fire_spread_multiple_points(self):
        initial_points = [(0, 0), (10, 10)]
        terrain = {"type": "forest", "density": 0.8}
        wind_dir = "E"
        wind_spd = 5.0
        time_steps = 2

        result = simulate_fire_spread(initial_points, terrain, wind_dir, wind_spd, time_steps)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        # Ожидаем 2 начальные точки * 2 временных шага = 4 элемента
        self.assertEqual(len(result), 4)

    def test_simulate_fire_spread_zero_time_steps(self):
        initial_points = [(0, 0)]
        terrain = {"type": "forest", "density": 0.8}
        wind_dir = "S"
        wind_spd = 0.0
        time_steps = 0

        result = simulate_fire_spread(initial_points, terrain, wind_dir, wind_spd, time_steps)
        self.assertEqual(len(result), 0)

    def test_simulate_fire_spread_no_initial_points(self):
        initial_points = []
        terrain = {"type": "forest", "density": 0.8}
        wind_dir = "W"
        wind_spd = 10.0
        time_steps = 5

        result = simulate_fire_spread(initial_points, terrain, wind_dir, wind_spd, time_steps)
        self.assertEqual(len(result), 0)

if __name__ == '__main__':
    unittest.main()
