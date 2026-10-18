"""Prepare geometric FOV/LOS evidence at an existing navigation pose.

This numerical preparation is separate from figure rendering. The sensor is
never scanned and no controller or map is rerun; all casts are geometry audits.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .. import api
from ..trajectory.engine import mounted_sensor_pose
from ..trajectory.storage import load_result
from .navigation_dependencies import TARGET_LABEL, emitted_rays, opportunity_audit, _scene


def _box_geometry(obj, role):
    local = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float)
    vertices = local * obj.h @ obj.R.T + obj.c
    faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    return dict(label=obj.label if hasattr(obj, "label") else role, role=role,
                vertices=vertices.tolist(), faces=faces,
                label_at=(obj.c + [0,0,obj.h[2]+.12]).tolist())


def _subset(indices, limit):
    indices = np.asarray(indices, dtype=int)
    return indices[np.unique(np.linspace(0, len(indices)-1, min(limit, len(indices))).astype(int))] if len(indices) else indices


def build_navigation_coverage(dependency_dir, output_dir):
    """Save paired absent/present-panel ray geometry using a saved baseline run."""
    directory, output = Path(dependency_dir).resolve(), Path(output_dir).resolve()
    baseline_path = directory / "runs" / "baseline" / "seed_1"
    result = load_result(baseline_path)
    absent = copy.deepcopy(result["config"])
    absent["_effective_sensor"].update(hfov_start_deg=-60., hfov_end_deg=60.)
    present = copy.deepcopy(absent)
    present["world"]["unexpected_objects"].append(dict(
        type="box", label="study_occluder", center=[9.7,1.1,.9], size=[.15,2.,1.8],
        yaw_deg=0., material={"rho_d":.3,"rho_s":0.}))
    scenes = [_scene(absent), _scene(present)]
    cfg = api.build_sensor(absent["_effective_sensor"])
    target_id = scenes[1].labels.index(TARGET_LABEL)
    panel_id = scenes[1].labels.index("study_occluder")
    target = scenes[1].objects[target_id]
    candidates = []
    for step in range(len(result["controls"]["t"])):
        pose = mounted_sensor_pose(absent, result["states"]["xy"][step], result["states"]["theta"][step])
        origin, direction = emitted_rays(cfg, pose)
        target_distance, _ = target.intersect(origin, direction)
        ranges, ids, _ = scenes[1].cast(origin, direction)
        nominal = np.isfinite(target_distance)
        visible = nominal & (ids == target_id)
        blocked_panel = nominal & (ids == panel_id)
        if visible.sum() and blocked_panel.sum():
            # Prefer an informative near-target view with comparable ray classes.
            distance = np.linalg.norm(np.asarray(pose.position)-target.c)
            balance = min(int(visible.sum()), int(blocked_panel.sum()))
            candidates.append((balance, -distance, step, pose))
    if not candidates:
        raise ValueError("No recorded baseline pose contains both visible and panel-blocked target rays")
    _, _, step, pose = max(candidates, key=lambda row: row[:2])
    origin, direction = emitted_rays(cfg, pose)
    target_range, _ = target.intersect(origin, direction)
    nominal = np.isfinite(target_range)
    pose_rotation = (api.rot_z(np.radians(pose.yaw_deg)) @ api.rot_y(np.radians(pose.pitch_deg))
                     @ api.rot_x(np.radians(pose.roll_deg)))
    local_direction = direction @ pose_rotation
    azimuth = np.degrees(np.arctan2(local_direction[:,1], local_direction[:,0]))
    elevation = np.degrees(np.arcsin(local_direction[:,2]))
    panel_records = []
    for index, (config, scene) in enumerate(zip((absent,present), scenes)):
        ranges, nearest_id, _ = scene.cast(origin, direction)
        tid = scene.labels.index(TARGET_LABEL)
        visible = nominal & (nearest_id == tid)
        blocked = nominal & ~visible
        selection = np.r_[_subset(np.flatnonzero(visible),6), _subset(np.flatnonzero(blocked),6)]
        rays = []
        for ray in selection:
            rays.append(dict(beam_index=int(ray), target_point=(origin[ray]+target_range[ray]*direction[ray]).tolist(),
                             nearest_point=(origin[ray]+ranges[ray]*direction[ray]).tolist(),
                             visible=bool(visible[ray]), nearest_label=scene.labels[int(nearest_id[ray])],
                             azimuth_deg=float(azimuth[ray]), elevation_deg=float(elevation[ray]),
                             target_range_m=float(target_range[ray]), nearest_range_m=float(ranges[ray]),
                             range_eligible=bool(cfg.min_range_m<=ranges[ray]<=cfg.max_range_m)))
        objects = [_box_geometry(scene.objects[tid],"target")]
        if index:
            objects.append(_box_geometry(scene.objects[scene.labels.index("study_occluder")],"occluder"))
        audit = opportunity_audit(cfg, pose, scene)
        panel_records.append(dict(id="panel_present" if index else "panel_absent",
                                  label="Opaque panel present" if index else "Opaque panel absent",
                                  objects=objects, rays=rays, audit=audit,
                                  displayed_visible_rays=int(visible[selection].sum()),
                                  displayed_blocked_rays=int(blocked[selection].sum())))
    origin_one = np.asarray(pose.position,float)
    distance = np.linalg.norm(origin_one-target.c)
    radius = min(8., max(distance+1.5,4.))
    payload = dict(schema_version=1,
                   source=dict(run=str(baseline_path.relative_to(directory)), seed=1, step=step,
                               time_s=float(result["states"]["t"][step]),
                               config="Saved cluttered navigation baseline with front-120° sensor coverage; panel is sole scene difference"),
                   sensor=dict(position=origin_one.tolist(),rotation_world_from_sensor=pose_rotation.tolist(),
                               yaw_deg=float(pose.yaw_deg),pitch_deg=float(pose.pitch_deg),roll_deg=float(pose.roll_deg)),
                   hfov_deg=[-60.,60.],vfov_deg=[float(min(cfg.elevations_deg())),float(max(cfg.elevations_deg()))],
                   min_range_m=float(cfg.min_range_m),max_range_m=float(cfg.max_range_m),display_range_m=float(radius),
                   panels=panel_records,
                   view_limits=[[float(origin_one[0]-.5),12.],[-1.8,4.3],[0.,2.5]],
                   view_elevation_deg=22,view_azimuth_deg=-64,
                   interpretation="Prepared scene truth and deterministic nearest-surface ray audits, not a noisy point cloud. The nominal FOV is angular and is truncated in radius only for display. All panel counts use every emitted ray; only a representative subset is drawn.")
    output.mkdir(parents=True,exist_ok=True)
    destination=output/"fov_los.json"
    destination.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    return dict(path=str(destination),source=payload["source"],
                audits={row["id"]:row["audit"] for row in panel_records})
