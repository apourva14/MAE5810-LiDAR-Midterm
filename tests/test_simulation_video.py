"""Video-specific protections for terminated histories and tested-case ranking."""
from pathlib import Path
import unittest

import numpy as np

from lidar_project.visualization.simulation_video import RecordedRun, SENSOR_CASES, select_sensor_extremes


class SimulationVideoTests(unittest.TestCase):
    def test_common_clock_freezes_terminated_run_without_future_scan_or_voxels(self):
        run=RecordedRun("test",{"status":"collision"},{
            "states_t":np.array([0.,.2,.4]),"controls_t":np.array([0.,.2]),
            "scan_offsets":np.array([0,1,2]),"scan_xyz_world":np.array([[0,0,0],[1,0,0]])},
            {"map_first_seen":np.array([0,1]),"ground_first_seen":np.array([[0,-1]]),
             "obstacle_first_seen":np.array([[-1,1]])},{},())
        early=run.at(.1)
        self.assertEqual(early["scan"],0)
        self.assertEqual(early["state"],0)
        self.assertEqual(early["map_voxels"],1)
        self.assertEqual(early["observed_fraction"],.5)
        self.assertFalse(early["ended"])
        terminal=run.at(.4); held=run.at(20.)
        self.assertTrue(held["ended"])
        self.assertEqual(held["state"],2)
        self.assertEqual(held["scan"],1)
        self.assertEqual(held["map_voxels"],2)
        np.testing.assert_array_equal(terminal["xyz"],held["xyz"])

    def test_disabled_sensor_cannot_be_selected_and_safety_precedes_coverage(self):
        cases=[]
        for index,case_id in enumerate(SENSOR_CASES):
            cases.append({"id":case_id,"aggregate":dict(goals=3,collisions=0,timeouts=0,
                worst_surface_gap_m=.1+index*.01,mean_observed_fraction=.99-index*.03)})
        cases.append({"id":"lidar_disabled","aggregate":dict(goals=0,collisions=3,timeouts=0,
            worst_surface_gap_m=-1,mean_observed_fraction=0)})
        best,worst,ordered=select_sensor_extremes({"cases":cases})
        self.assertEqual(best["id"],SENSOR_CASES[-1])
        self.assertEqual(worst["id"],SENSOR_CASES[0])
        self.assertEqual(len(ordered),10)
        with self.assertRaisesRegex(ValueError,"all ten"):
            select_sensor_extremes({"cases":cases[1:]})


if __name__=="__main__":
    unittest.main()
