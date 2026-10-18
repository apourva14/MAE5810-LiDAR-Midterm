"""Behavioral references captured from the original simulator before reorganization."""
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from lidar_project import api
from lidar_project.navigation.engine import run
from lidar_project.navigation.storage import load_results, save_results, require_frames
from lidar_project.cli import load_navigation_config
from lidar_project.paths import PROJECT_ROOT

FIXTURES = Path(__file__).parent / "fixtures"


class SensorBehavior(unittest.TestCase):
    def test_original_seeded_scan_references(self):
        for path in sorted(FIXTURES.glob("sensor_*.npz")):
            with self.subTest(case=path.stem), np.load(path, allow_pickle=False) as reference:
                cfg, pose, scene, sim, _ = api.build_all(json.loads(str(reference["config_json"])))
                cloud = api.LidarSensor(cfg, pose, seed=sim["seed"]).scan(scene)
                for key in cloud.__dataclass_fields__:
                    actual, expected = np.asarray(getattr(cloud, key)), reference[key]
                    if expected.dtype.kind in "US":
                        np.testing.assert_array_equal(actual, expected)
                    else:
                        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_saved_scan_round_trip(self):
        path = FIXTURES / "sensor_moving.npz"
        cloud = api.load_npz(path)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "scan.npz"
            cloud.save_npz(destination)
            loaded = api.load_npz(destination)
            for key in cloud.__dataclass_fields__:
                np.testing.assert_array_equal(np.asarray(getattr(loaded, key)), np.asarray(getattr(cloud, key)))


class NavigationBehavior(unittest.TestCase):
    def test_original_navigation_references(self):
        for path in sorted(FIXTURES.glob("navigation_*.npz")):
            with self.subTest(case=path.stem), np.load(path, allow_pickle=False) as reference:
                nav = json.loads(str(reference["nav_json"]))
                scenario = json.loads(str(reference["scenario_json"]))
                result = run(nav, scenario, verbose=False)
                np.testing.assert_allclose(result["log"], reference["log"], rtol=1e-12, atol=1e-12, equal_nan=True)
                self.assertEqual(result["status"], str(reference["status"]))
                expected = json.loads(str(reference["first_det_json"]))
                self.assertEqual(list(result["first_det"]), list(expected))
                np.testing.assert_allclose(list(result["first_det"].values()), list(expected.values()), equal_nan=True)

    def test_navigation_round_trip_preserves_replay(self):
        with np.load(FIXTURES / "navigation_clear.npz", allow_pickle=False) as reference:
            nav = json.loads(str(reference["nav_json"]))
            scenario = json.loads(str(reference["scenario_json"]))
        result = run(nav, scenario, verbose=False)
        with tempfile.TemporaryDirectory() as temporary:
            save_results([result], nav, temporary)
            loaded, restored_nav = load_results(temporary)
            self.assertEqual(restored_nav["robot"], nav["robot"])
            self.assertEqual(len(loaded), 1)
            restored = loaded[0]
            np.testing.assert_array_equal(restored["log"], result["log"])
            require_frames(restored)
            self.assertEqual(len(restored["history"]), len(result["history"]))
            for old, new in zip(result["history"], restored["history"]):
                self.assertEqual(old["k"], new["k"])
                np.testing.assert_array_equal(old["pc"].xyz_world, new["pc"].xyz_world)

    def test_historical_logs_have_no_invented_frames(self):
        results, nav = load_results(FIXTURES / "historical")
        self.assertEqual({r["name"] for r in results}, {"clear", "heavy_rain", "dense_fog"})
        for result in results:
            self.assertFalse(result["replay_available"])
            with self.assertRaises((ValueError, FileNotFoundError)):
                require_frames(result)


class ModuleBoundaries(unittest.TestCase):
    def test_numerical_commands_need_no_visualization(self):
        # Refuse optional presentation imports even if installed on this machine.
        script = '''
import importlib.abc, sys
class BlockVisuals(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib','plotly','PIL','tkinter'} or fullname.startswith('lidar_project.visualization'):
            raise RuntimeError('Numerical layer imported visualization: '+fullname)
sys.meta_path.insert(0, BlockVisuals())
from lidar_project.cli import main
from lidar_project import api
from lidar_project.navigation.engine import run
from lidar_project.navigation.storage import load_results
assert main(['scan','--set','sensor.n_channels=2','--set','sensor.measurements_per_cycle=24','--set','sensor.n_beam_samples=1','--data-dir',sys.argv[1]]) == 0
assert main(['navigate','--scenario','clear','--max-steps','2','--data-dir',sys.argv[2]]) == 0
assert main(['trajectory','--max-steps','2','--data-dir',sys.argv[3]]) == 0
from lidar_project.trajectory import load_result
from lidar_project.mapping import build_map
mapped = build_map(load_result(sys.argv[3]))
assert mapped['metadata']['steps'] == 2
'''
        with tempfile.TemporaryDirectory() as temporary:
            finished = subprocess.run([sys.executable, "-c", script, str(Path(temporary)/"scan"),
                                       str(Path(temporary)/"nav"), str(Path(temporary)/"trajectory")],
                                      cwd=PROJECT_ROOT, capture_output=True, text=True)
            self.assertEqual(finished.returncode, 0, finished.stdout + finished.stderr)

    def test_config_resolution_does_not_depend_on_working_directory(self):
        nav = load_navigation_config()
        self.assertTrue(Path(nav["base_config"]).is_file())
        self.assertIn("sensor", nav["_base_config_snapshot"])


if __name__ == "__main__":
    unittest.main()
