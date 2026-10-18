"""Run-level inference and failure-aware paired navigation summaries."""
import copy
import unittest

import numpy as np

from lidar_project.experiments.environments import environment_cases
from lidar_project.experiments.navigation_robustness import (
    aggregate_runs, describe, paired_differences, robustness_cases, wilson_interval, _slow_episodes,
)
from lidar_project.trajectory import load_config
from lidar_project.trajectory.configuration import effective_config


def metric(seed, goal=True, gap=.3, elapsed=15., path=19.):
    return dict(seed=seed, status="goal" if goal else "collision", goal_completed=int(goal),
                collision=int(not goal), timeout=0, elapsed_s=elapsed, path_length_m=path,
                swept_min_surface_gap_m=gap, spatial_route_rmse_m=.4,
                mean_returns_per_control_step=1000.)


class NavigationRobustness(unittest.TestCase):
    def test_wilson_uses_runs_and_has_nonzero_width_at_boundary(self):
        low, high = wilson_interval(20, 20)
        self.assertAlmostEqual(low, .8388748419471806)
        self.assertAlmostEqual(high, 1.)
        low0, high0 = wilson_interval(0, 20)
        self.assertAlmostEqual(low0, 0.)
        self.assertAlmostEqual(high0, 1 - low)
        with self.assertRaises(ValueError):
            wilson_interval(1, 0)
        with self.assertRaises(ValueError):
            wilson_interval(21, 20)

    def test_failures_are_counted_but_not_treated_as_fast_successes(self):
        rows = [metric(1), metric(2, goal=False, gap=-.1, elapsed=3., path=2.)]
        summary = aggregate_runs(rows)
        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["goals"], 1)
        self.assertEqual(summary["collisions"], 1)
        self.assertEqual(summary["goal_fraction"], .5)
        self.assertEqual(summary["all_run_prefix"]["elapsed_s"]["median"], 9.)
        self.assertEqual(summary["all_run_prefix"]["swept_min_surface_gap_m"]["min"], -.1)
        self.assertEqual(summary["successful_runs"]["elapsed_s"]["median"], 15.)
        failed = aggregate_runs([rows[1]])
        self.assertEqual(failed["successful_runs"]["elapsed_s"]["n"], 0)
        self.assertIsNone(failed["successful_runs"]["elapsed_s"]["median"])

    def test_paired_time_conditions_on_both_successes(self):
        baseline = [metric(2, elapsed=12.), metric(1, elapsed=15.)]
        active = [metric(1, elapsed=17.), metric(2, goal=False, gap=-.1, elapsed=3.)]
        paired = paired_differences(active, baseline)
        self.assertEqual(paired["pairs"], 2)
        self.assertEqual(paired["baseline_only_goal"], 1)
        self.assertEqual(paired["both_goal"], 1)
        self.assertEqual(paired["goal_fraction_difference"], -.5)
        self.assertEqual(paired["both_success_deltas"]["elapsed_s"]["median"], 2.)
        self.assertEqual(paired["per_seed"][0]["seed"], 1)
        self.assertEqual(paired["per_seed"][1]["delta_elapsed_s"], -9.)
        with self.assertRaises(ValueError):
            paired_differences(active, baseline[:1])
        with self.assertRaises(ValueError):
            paired_differences([active[0], active[0]], baseline)

    def test_summary_iqr_is_explicit_and_nonfinite_values_rejected(self):
        result = describe([1., 2., 3., 4.])
        self.assertEqual(result["median"], 2.5)
        self.assertEqual(result["q25"], 1.75)
        self.assertEqual(result["q75"], 3.25)
        self.assertIsNone(describe([])["mean"])
        with self.assertRaises(ValueError):
            describe([np.nan])

    def test_low_speed_episodes_use_resulting_state_time_and_retain_final_interval(self):
        result = dict(states={"t": np.arange(7)*.2, "xy": np.zeros((7,2))},
                      controls={"v": np.array([0., 0., 1., .01, .01, .01]),
                                "n_returns": np.arange(6), "n_obstacle_points": np.arange(6),
                                "omega": np.zeros(6)})
        episodes = _slow_episodes(result, minimum_steps=2)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]["first_control_step"], 0)
        self.assertEqual(episodes[0]["stop_control_step_exclusive"], 2)
        self.assertAlmostEqual(episodes[0]["end_time_s"], .4)
        self.assertEqual(episodes[1]["first_control_step"], 3)
        self.assertEqual(episodes[1]["stop_control_step_exclusive"], 6)
        self.assertAlmostEqual(episodes[1]["duration_s"], .6)
        self.assertAlmostEqual(episodes[1]["mean_returns"], 4.)

    def test_cases_freeze_geometry_controller_robot_and_have_matched_side_fov(self):
        cases = robustness_cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case["id"] for case in cases}), 12)
        self.assertEqual(next(case for case in cases if case["id"] == "fov_side")["comparison_baseline"], "fov_front")
        self.assertEqual([c["overrides"]["sensor"]["air_extinction_per_m"] for c in cases if c["factor"] == "optical_stress"], [.1, .3, .6, 1.2, 3., 10.])
        baseline_config = load_config()
        untouched = copy.deepcopy(baseline_config)
        layouts = environment_cases(baseline_config)
        self.assertEqual(baseline_config, untouched)
        self.assertEqual([layout["id"] for layout in layouts], ["open", "cluttered", "narrow"])
        for layout in layouts:
            baseline = layout["config"]
            for case in cases:
                effective, _ = effective_config(baseline, {"name": case["id"], **case["overrides"]})
                self.assertEqual(effective["world"], baseline["world"])
                self.assertEqual(effective["robot"], baseline["robot"])
                self.assertEqual(effective["controller"], baseline["controller"])
                self.assertEqual(effective["planner"], baseline["planner"])
        fronts = next(case for case in cases if case["id"] == "fov_front")
        side = next(case for case in cases if case["id"] == "fov_side")
        self.assertEqual(fronts["overrides"]["sensor"], side["overrides"]["sensor"])


if __name__ == "__main__":
    unittest.main()
