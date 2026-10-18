"""Checks for the study's timing/spatial distinction and failure denominators."""
import unittest
from pathlib import Path

import numpy as np

from lidar_project.experiments.trajectory import (
    evaluate_result, factor_matrix, point_polyline_distances, summarize_cases, tracking_metrics,
)


class TrajectoryEvaluationTests(unittest.TestCase):
    def test_spatial_and_clock_errors_are_different(self):
        reference = {"xy": np.array([[0., 0.], [2., 0.]]), "t": np.array([0., 1.])}
        positions = np.array([[0., 0.], [1., 0.], [2., 0.]])
        metrics = tracking_metrics(np.array([0., 1., 2.]), positions, reference)
        self.assertEqual(metrics["spatial_route_rmse_m"], 0.)
        self.assertAlmostEqual(metrics["time_reference_rmse_m"], 1 / np.sqrt(3))
        self.assertEqual(metrics["time_reference_max_m"], 1.)

    def test_terminal_state_contributes_to_error(self):
        reference = {"xy": np.array([[0., 0.], [2., 0.]]), "t": np.array([0., 2.])}
        positions = np.array([[0., 0.], [1., 0.], [2., 3.]])
        metrics = tracking_metrics(np.array([0., 1., 2.]), positions, reference)
        self.assertAlmostEqual(metrics["spatial_route_rmse_m"], np.sqrt(3))
        self.assertEqual(metrics["spatial_route_max_m"], 3.)
        self.assertAlmostEqual(metrics["time_reference_rmse_m"], np.sqrt(3))

    def test_distance_uses_segments_not_only_waypoints(self):
        distances = point_polyline_distances([[1., 1.], [3., 0.]], [[0., 0.], [2., 0.]])
        np.testing.assert_array_equal(distances, [1., 1.])

    def test_failures_remain_in_summary_denominators(self):
        common = dict(case_id="baseline", factor="baseline", path_length_m=2.,
                      spatial_route_rmse_m=.1, spatial_route_max_m=.2,
                      time_reference_rmse_m=.3, angular_rate_total_variation_rad_s=.4,
                      mean_returns_per_control_step=12.)
        first = dict(common, status="goal", goal_completed=1, collision=0, timeout=0,
                     elapsed_s=4., swept_min_surface_gap_m=.5)
        second = dict(common, status="collision", goal_completed=0, collision=1, timeout=0,
                      elapsed_s=2., swept_min_surface_gap_m=-.1)
        summary = summarize_cases([first, second], factor_matrix())[0]
        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["goals"], 1)
        self.assertEqual(summary["collisions"], 1)
        self.assertEqual(summary["goal_fraction"], .5)
        self.assertEqual(summary["mean_termination_time_s_all_runs"], 3.)
        self.assertEqual(summary["min_surface_gap_m"], -.1)

    def test_factor_matrix_changes_one_physical_factor(self):
        cases = factor_matrix()
        self.assertEqual(len(cases), 7)
        self.assertEqual(cases[0]["overrides"], {})
        for case in cases[1:]:
            changes = case["overrides"]
            self.assertEqual(len(changes), 1)
            if isinstance(next(iter(changes.values())), dict):
                self.assertEqual(len(next(iter(changes.values()))), 1)
        # This must not change the known-map planner or planned reference.
        margin = next(case for case in cases if case["case_id"] == "smaller_margin")
        self.assertEqual(set(margin["overrides"]), {"controller"})

    def _simple_execution(self, obstacle_x):
        from lidar_project.trajectory import load_config, run_trajectory
        config = load_config(Path(__file__).resolve().parents[1] / "configs" / "trajectory.jsonc")
        config["robot"].update(start=[0., 0.], goal=[3., 0.], start_heading_deg=0.,
                               v_max=10., dt=.2, max_steps=1)
        config["world"]["known_objects"] = []
        config["world"]["unexpected_objects"] = [
            {"type": "box", "label": "test_obstacle", "center": [obstacle_x, 0., .5],
             "size": [.2, .2, 1.], "material": "concrete"}]
        xy = np.array([[0., 0.], [3., 0.]])
        reference = dict(xy=xy, raw_xy=xy.copy(), s=np.array([0., 3.]), t=np.array([0., 3.75]),
                         speed=np.array([.8, .8]), heading=np.zeros(2))
        return run_trajectory(config, {"name": "test", "lidar_enabled": False},
                              seed=1, record_history=False, reference=reference)

    def test_initial_collision_has_one_state_and_no_controls(self):
        result = self._simple_execution(0.)
        metric = evaluate_result(result, "test", "initial_overlap", 1, lidar_enabled=False)
        self.assertEqual(metric["status"], "collision")
        self.assertEqual(metric["collision"], 1)
        self.assertEqual(metric["state_count"], 1)
        self.assertEqual(metric["control_steps"], 0)
        self.assertEqual(metric["elapsed_s"], 0.)
        self.assertLess(metric["initial_surface_gap_m"], 0.)

    def test_swept_collision_counts_when_endpoints_are_clear(self):
        result = self._simple_execution(1.)
        self.assertTrue(np.all(result["states"]["surface_gap"] > 0))
        metric = evaluate_result(result, "test", "between_states", 1, lidar_enabled=False)
        self.assertEqual(metric["status"], "collision")
        self.assertLess(metric["swept_min_surface_gap_m"], 0.)
        self.assertEqual(metric["state_count"], 2)
        self.assertEqual(metric["control_steps"], 1)
        self.assertAlmostEqual(metric["elapsed_s"], .2)
        self.assertAlmostEqual(metric["path_length_m"], 2.)


if __name__ == "__main__":
    unittest.main()
