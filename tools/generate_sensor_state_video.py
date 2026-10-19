#!/usr/bin/env python3
"""Generate one synchronized best / worst enabled sensor-configuration video.

Run from the repository: python tools/generate_sensor_state_video.py
Ranks all ten tested fixed-scene, clear-air sensor cases by task completion and
safety before endpoint-cell coverage. Missing inputs are regenerated separately
in data/video_inputs/sensor_states; frozen report studies are never replaced.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_project.visualization.simulation_video import video_main

if __name__ == "__main__":
    video_main("sensor")
