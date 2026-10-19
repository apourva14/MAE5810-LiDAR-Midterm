"""Recorded 3D LiDAR movies, independent of numerical experiment execution.

The video clock uses actual simulation seconds. Scans and endpoint-map voxels
appear only at their saved control time; the terminal robot state has no scan.
Rendering downsamples displayed voxels deterministically, never metric counts.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
SENSOR_CASES = ("baseline", "mount_low", "mount_high", "pitch_up", "pitch_down",
                "yaw_full", "fov_front", "fov_side", "noise_high", "sampling_sparse")
RANKING_RULE = ["maximize completed goals", "minimize collisions", "minimize timeouts",
                "maximize worst swept surface gap", "maximize mean observed endpoint-cell fraction"]
PALETTE = ("#2875A2", "#8B68B0", "#38988D")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_sensor_extremes(summary):
    """Rank all ten tested, enabled, fixed-scene/clear-air sensor cases.

    Safety and task completion take priority over endpoint-map coverage. This is
    a descriptive ordering of tested configurations, not a universal optimum.
    """
    cases = {c["id"]: c for c in summary["cases"]}
    absent = set(SENSOR_CASES) - cases.keys()
    if absent:
        raise ValueError(f"Sensor-state ranking needs all ten cases: missing {sorted(absent)}")
    def key(case):
        a = case["aggregate"]
        return (-a["goals"], a["collisions"], a["timeouts"],
                -a["worst_surface_gap_m"], -a["mean_observed_fraction"], case["id"])
    ordered = sorted((cases[c] for c in SENSOR_CASES), key=key)
    return ordered[0], ordered[-1], ordered


@dataclass
class RecordedRun:
    case_id: str
    metadata: dict
    arrays: dict
    mapping: dict
    map_metadata: dict
    source_paths: tuple[Path, ...]

    @property
    def end_time(self):
        return float(self.arrays["states_t"][-1])

    def at(self, time_s):
        """Causal snapshot; never extrapolate motion, scans or measured cells."""
        d, m = self.arrays, self.mapping
        state = min(len(d["states_t"]) - 1,
                    int(np.searchsorted(d["states_t"], time_s + 1e-9, side="right") - 1))
        scan = min(len(d["controls_t"]) - 1,
                   int(np.searchsorted(d["controls_t"], time_s + 1e-9, side="right") - 1))
        if state < 0:
            raise ValueError("Video time cannot precede the recorded initial state")
        if scan < 0:
            xyz = np.empty((0, 3))
        else:
            offsets = d["scan_offsets"]
            xyz = d["scan_xyz_world"][offsets[scan]:offsets[scan + 1]]
        map_mask = (m["map_first_seen"] >= 0) & (m["map_first_seen"] <= scan)
        seen = ((m["ground_first_seen"] >= 0) & (m["ground_first_seen"] <= scan)
                | (m["obstacle_first_seen"] >= 0) & (m["obstacle_first_seen"] <= scan))
        return dict(state=state, scan=scan, xyz=xyz, map_mask=map_mask,
                    observed_fraction=float(seen.mean()), observed_cells=int(seen.sum()),
                    map_voxels=int(map_mask.sum()), ended=bool(time_s >= self.end_time - 1e-9))


def load_recording(directory, case_id, seed=1):
    run = Path(directory) / "runs" / case_id / f"seed_{seed}"
    paths = tuple(run / name for name in ("trajectory.json", "trajectory.npz", "mapping.json", "mapping.npz"))
    metadata, map_metadata = _read(paths[0]), _read(paths[2])
    with np.load(paths[1], allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    with np.load(paths[3], allow_pickle=False) as archive:
        mapping = {k: archive[k].copy() for k in archive.files}
    k = len(arrays["controls_t"])
    if (not metadata.get("scans_available") or len(arrays["states_t"]) != k + 1
            or len(arrays["scan_offsets"]) != k + 1
            or not np.allclose(arrays["controls_t"], arrays["states_t"][:-1])):
        raise ValueError(f"Video requires aligned raw 3D scan histories: {run}")
    if not np.array_equal(np.diff(arrays["scan_offsets"]), arrays["controls_n_returns"]):
        raise ValueError(f"Raw scan counts do not match recorded metrics: {run}")
    if np.any(mapping["map_first_seen"] < 0) or np.any(mapping["map_first_seen"] >= k):
        raise ValueError(f"Map contains an invalid scan index: {run}")
    return RecordedRun(case_id, metadata, arrays, mapping, map_metadata, paths)


def _same_scene(runs):
    first = runs[0]
    world = json.dumps(first.metadata["world_objects"], sort_keys=True)
    reference = first.arrays["reference_xy"]
    for run in runs[1:]:
        if json.dumps(run.metadata["world_objects"], sort_keys=True) != world:
            raise ValueError("Video comparison must use identical obstacle and target geometry/material")
        if not np.array_equal(run.arrays["reference_xy"], reference):
            raise ValueError("Video comparison must use an identical planning reference")


def _faces(obj):
    """Display-only tessellation of stored geometry; no ray casting."""
    kind = obj["type"]
    if kind == "box":
        vertices = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                             [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float) * np.asarray(obj["size"]) / 2
        angle = np.radians(obj.get("yaw_deg", 0))
        rotation = np.array([[np.cos(angle),-np.sin(angle),0], [np.sin(angle),np.cos(angle),0], [0,0,1]])
        vertices = vertices @ rotation.T + obj["center"]
        return [vertices[i] for i in ([0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7])]
    if kind == "cylinder":
        p = np.linspace(0, 2*np.pi, 17)
        ring = np.c_[obj["radius"]*np.cos(p),obj["radius"]*np.sin(p),np.zeros(len(p))] + obj["base"]
        top = ring + [0,0,obj["height"]]
        return [ring[:-1],top[:-1]] + [np.array([ring[i],ring[i+1],top[i+1],top[i]]) for i in range(16)]
    if kind == "sphere":
        t, p = np.linspace(0,np.pi,7), np.linspace(0,2*np.pi,13)
        v = np.array([[np.sin(a)*np.cos(b),np.sin(a)*np.sin(b),np.cos(a)] for a in t for b in p]) * obj["radius"] + obj["center"]
        return [v[[i*13+j,i*13+j+1,(i+1)*13+j+1,(i+1)*13+j]] for i in range(6) for j in range(12)]
    return []


def _rotation(roll, pitch, yaw):
    r,p,y = np.radians([roll,pitch,yaw])
    rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    return rz @ ry @ rx


def _sensor_glyph(run, xy, theta):
    """Nominal angular FOV drawn to 2.3 m only; glyph is not measured range."""
    config, sensor = run.metadata["config"], run.metadata["sensor"]
    mount = config.get("sensor_mount", {})
    heading = np.degrees(theta) + mount.get("yaw_deg", 0)
    rotation = _rotation(mount.get("roll_deg", 0), mount.get("pitch_deg", 0), heading)
    offset = np.asarray(mount.get("offset_xy_m", [0,0]), float)
    robot_rotation = _rotation(0,0,np.degrees(theta))
    origin = np.r_[xy, config["mount_height_m"]] + np.r_[offset,0] @ robot_rotation.T
    az = np.linspace(sensor["hfov_start_deg"], sensor["hfov_end_deg"], 37)
    boundaries = []
    for elevation in (sensor["vfov_lower_deg"],sensor["vfov_upper_deg"]):
        a,e = np.radians(az), np.radians(elevation)
        local = np.c_[np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.full(len(a),np.sin(e))]
        boundaries.append(origin + 2.3*(local @ rotation.T))
    segments = [boundaries[0],boundaries[1]]
    for i in (0,len(az)//4,len(az)//2,3*len(az)//4,len(az)-1):
        segments.extend([np.vstack([origin,boundaries[0][i]]),np.vstack([origin,boundaries[1][i]])])
    return segments, origin


def _source_manifest(runs, summary_path):
    files = [summary_path] + [path for run in runs for path in run.source_paths]
    result = []
    for path in files:
        try:
            label = path.resolve().relative_to(PROJECT).as_posix()
        except ValueError:
            label = path.name
        result.append({"path":label,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    return result


def _frame_axes(fig, run, index, count, color):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    left, width = .035 + index*.93/count, .91/count
    ax = fig.add_axes([left-.005,.285,width+.005,.525],projection="3d")
    ax.set_facecolor("#F7F9FC")
    ax.set(xlim=(-2,21),ylim=(-6,7),zlim=(-.2,3.2))
    ax.set_box_aspect((23,13,5.2),zoom=1.06)
    ax.view_init(elev=47,azim=-102)
    ax.set_xticks([0,5,10,15,20]); ax.set_yticks([-5,0,5]); ax.set_zticks([0,2])
    ax.set_xlabel("x (m)",labelpad=-1);ax.set_ylabel("y (m)",labelpad=-3);ax.set_zlabel("z (m)",labelpad=-3)
    ax.tick_params(labelsize=8,pad=-1)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.fill=False
        axis._axinfo["grid"]["color"]=(.78,.82,.87,.55)
        axis._axinfo["grid"]["linewidth"]=.45
    for obj in run.metadata["world_objects"]:
        faces = _faces(obj)
        if faces:
            target = obj.get("label") == "study_target"
            ax.add_collection3d(Poly3DCollection(faces,facecolor="#B1A597" if target else "#AEB8C4",
                                               edgecolor="#887766" if target else "#8995A3",alpha=.12,
                                               linewidth=.5))
    reference=run.arrays["reference_xy"]
    ax.plot(reference[:,0],reference[:,1],np.full(len(reference),.025),c="#9AA5B3",ls=(0,(5,3)),lw=1.1)
    goal=run.metadata["config"]["robot"]["goal"]
    ax.scatter([goal[0]],[goal[1]],[.08],marker="*",s=110,c="#294352",depthshade=False)
    ax.text(goal[0]+.1,goal[1]+.4,.08,"Goal",fontsize=9,color="#294352")
    cloud=ax.scatter([],[],[],s=.85,c=color,alpha=.48,depthshade=False,linewidths=0)
    scan=ax.scatter([],[],[],s=1.7,c="#E4892E",alpha=.88,depthshade=False,linewidths=0)
    path,=ax.plot([],[],[],c=color,lw=2.1)
    body=ax.scatter([],[],[],s=80,c="#173C59",edgecolors="white",linewidths=.8,depthshade=False)
    heading,=ax.plot([],[],[],c="#173C59",lw=2.4)
    glyph=Line3DCollection([np.zeros((2,3))],colors="#319C73",linewidths=.6,alpha=.52)
    ax.add_collection3d(glyph)
    sensor_point=ax.scatter([],[],[],s=23,c="#319C73",edgecolors="white",linewidths=.5,depthshade=False)
    return dict(ax=ax,cloud=cloud,scan=scan,path=path,body=body,heading=heading,glyph=glyph,
                sensor=sensor_point,left=left,width=width)


def render_comparison(runs, labels, parameter_lines, output, *, title, subtitle, summary,
                      summary_path, mode, selection=None, fps=15, width=1600, height=900,
                      hold_seconds=3., ffmpeg=None, display_voxels=12000):
    """Encode a synchronized movie of stored snapshots with ffmpeg/H.264."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _same_scene(runs)
    if width%2 or height%2 or width<800 or height<450 or fps<5 or hold_seconds<0:
        raise ValueError("Use even dimensions >=800x450, fps >=5 and nonnegative hold time")
    executable=ffmpeg or shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg is required (macOS: brew install ffmpeg)")
    output=Path(output).expanduser().resolve();output.parent.mkdir(parents=True,exist_ok=True)
    qa=PROJECT/"build"/"video_qa"/output.stem;qa.mkdir(parents=True,exist_ok=True)
    count=len(runs); end=max(run.end_time for run in runs)
    # The numerical simulation records at 5 Hz. Encode repeated frames at 15 fps;
    # no invented intermediate measurements or time-normalized routes are added.
    dt=round(min(float(np.min(np.diff(run.arrays["controls_t"]))) for run in runs),9)
    snapshot_rate=1/dt
    frame_count=int(np.ceil((end+hold_seconds)*snapshot_rate-1e-8))
    fig=plt.figure(figsize=(width/100,height/100),dpi=100,facecolor="#F7F9FC")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"text.color":"#233547", "axes.labelcolor":"#576777"})
    fig.text(.04,.948,title,fontsize=25,fontweight="bold",color="#19394F")
    fig.text(.04,.911,subtitle,fontsize=11,color="#627485")
    clock=fig.text(.96,.947,"",ha="right",fontsize=20,fontweight="bold",color="#19394F")
    panels=[]
    for i,(run,label,params) in enumerate(zip(runs,labels,parameter_lines)):
        panel=_frame_axes(fig,run,i,count,PALETTE[i]);left,w=panel["left"],panel["width"]
        fig.text(left,.855,label,fontsize=19,fontweight="bold",color=PALETTE[i])
        fig.text(left,.823,params,fontsize=10.2,color="#5D6C7D",linespacing=1.45)
        panel["status"]=fig.text(left,.293,"",fontsize=11.7,fontweight="bold")
        panel["instant"]=fig.text(left,.249,"",fontsize=11,color="#40576A",linespacing=1.6)
        panel["totals"]=fig.text(left,.19,"",fontsize=11,color="#40576A",linespacing=1.65)
        a=next(c for c in summary["cases"] if c["id"]==run.case_id)["aggregate"]
        text=f"Study: {a['goals']}/{a['runs']} goals  |  {a['collisions']} collisions\nMean mapped endpoint cells: {100*a['mean_observed_fraction']:.1f}%"
        if mode=="sensor":
            text+=f"  |  min gap {a['worst_surface_gap_m']:.3f} m"
        fig.text(left,.126,text,fontsize=10,color="#718091",linespacing=1.7)
        m=run.mapping["map_xyz"]
        panel["display_index"]=np.linspace(0,len(m)-1,min(len(m),display_voxels),dtype=int)
        panels.append(panel)
    fig.text(.04,.065,"BLUE / PURPLE / TEAL  accumulated measured endpoints     ORANGE  latest scan     GREEN  nominal FOV (2.3 m glyph)",fontsize=9.2,color="#586E81")
    if mode=="environment":
        note="Weather changes optical attenuation only. No atmospheric backscatter / false weather returns. Same obstacles, target and sensor."
    else:
        note="Among 10 tested enabled configurations: goals, collisions, timeouts, minimum gap, then endpoint-cell coverage. Same clear-air scene."
    fig.text(.04,.037,note,fontsize=9.2,color="#586E81")
    fig.text(.96,.012,f"Recorded seed {runs[0].metadata['seed']}  |  exact simulator poses  |  display-only voxel thinning / fixed scene crop  |  native snapshots: {snapshot_rate:g} Hz",ha="right",fontsize=8,color="#8B98A7")
    command=[executable,"-hide_banner","-loglevel","error","-y","-f","rawvideo","-vcodec","rawvideo","-pix_fmt","rgb24",
             "-s",f"{width}x{height}","-framerate",f"{snapshot_rate:.8g}","-i","pipe:0","-an","-c:v","libx264",
             "-preset","fast","-crf","20","-pix_fmt","yuv420p","-r",str(fps),"-movflags","+faststart",str(output)]
    with (qa/"ffmpeg.log").open("wb") as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for frame in range(frame_count):
                time_s=min(frame/snapshot_rate,end)
                clock.set_text(f"t = {time_s:04.1f} s")
                for run,panel in zip(runs,panels):
                    s=run.at(time_s);d=run.arrays;k=s["state"];xy=d["states_xy"][k];theta=d["states_theta"][k]
                    index=panel["display_index"];index=index[s["map_mask"][index]]
                    points=run.mapping["map_xyz"][index]
                    panel["cloud"]._offsets3d=points.T if len(points) else ([],[],[])
                    p=s["xyz"]
                    crop=(p[:,0]>=-2)&(p[:,0]<=21)&(p[:,1]>=-6)&(p[:,1]<=7)&(p[:,2]>=-.2)&(p[:,2]<=3.2)
                    shown=p[crop];panel["scan"]._offsets3d=shown.T if len(shown) else ([],[],[])
                    path=d["states_xy"][:k+1];panel["path"].set_data_3d(path[:,0],path[:,1],np.full(len(path),.09))
                    panel["body"]._offsets3d=([xy[0]],[xy[1]],[.19])
                    arrow=np.array([np.r_[xy,.22],np.r_[xy+.85*np.array([np.cos(theta),np.sin(theta)]),.22]])
                    panel["heading"].set_data_3d(*arrow.T)
                    segments,origin=_sensor_glyph(run,xy,theta);panel["glyph"].set_segments(segments)
                    panel["sensor"]._offsets3d=([origin[0]],[origin[1]],[origin[2]])
                    status=run.metadata["status"]
                    if s["ended"]:
                        text=f"{'GOAL REACHED' if status=='goal' else status.upper()} at {run.end_time:.1f} s  |  recording ended"
                    else:
                        text="NAVIGATING  |  measurements update every 0.2 s"
                    panel["status"].set_text(text)
                    panel["status"].set_color("#AE473E" if s["ended"] and status=="collision" else "#237B63" if s["ended"] else "#40576A")
                    scan_time=float(d["controls_t"][s["scan"]]) if s["scan"]>=0 else 0
                    panel["instant"].set_text(f"{'Last scan' if s['ended'] else 'Latest scan'}: {len(p):,} points at {scan_time:.1f} s" + ("  (frozen)" if s["ended"] else ""))
                    panel["totals"].set_text(f"Accumulated map: {s['map_voxels']:,} measured voxels\nObserved endpoint cells: {s['observed_cells']:,} / {s['observed_fraction']*100:.1f}%")
                fig.canvas.draw()
                if frame in (0,int(end*snapshot_rate/2),int(round(end*snapshot_rate))):
                    fig.savefig(qa/f"frame_{frame:04d}.png",dpi=100)
                rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
                process.stdin.write(rgb.tobytes())
                if frame%max(1,frame_count//10)==0:
                    print(f"{output.name}: rendered {frame+1}/{frame_count} simulation snapshots",flush=True)
            process.stdin.close()
            if process.wait()!=0:
                raise RuntimeError((qa/"ffmpeg.log").read_text())
        except BaseException:
            process.kill();process.wait();raise
        finally:
            plt.close(fig)
    metadata={"format_version":1,"video":output.name,"mode":mode,"frame_size":[width,height],"encoded_fps":fps,
              "simulation_snapshot_hz":snapshot_rate,"rendered_snapshot_count":frame_count,
              "duration_s":frame_count/snapshot_rate,"common_time_axis_s":[0,end],"end_hold_s":hold_seconds,
              "chronology":"scan k at controls_t[k] = states_t[k]; terminal state has no scan; ended runs freeze without new measurements",
              "map_semantics":"first measured endpoint per 0.12 m voxel; observed XY cells are measured-endpoint cells, not free space or true map accuracy",
              "display":f"small endpoint markers; at most {display_voxels} evenly selected recorded map voxels per panel; latest scans are visually cropped to x[-2,21], y[-6,7], z[-0.2,3.2] m; all raw returns retained in metrics; static translucent meshes are truth context",
              "limitations":["exact simulator poses, static objects, ideal snapshot scans; no SLAM", "illustrative attenuation coefficients; no calibrated real-weather validation", "rain/fog attenuation only; no weather particles/backscatter"],
              "ranking":selection,"source_files":_source_manifest(runs,Path(summary_path)),"cases":[]}
    for run in runs:
        case=next(c for c in summary["cases"] if c["id"]==run.case_id)
        metadata["cases"].append(dict(id=run.case_id,seed=run.metadata["seed"],status=run.metadata["status"],
            end_time_s=run.end_time,last_scan_time_s=float(run.arrays["controls_t"][-1]),sensor=run.metadata["sensor"],
            sensor_mount=run.metadata["config"].get("sensor_mount"),mount_height_m=run.metadata["config"]["mount_height_m"],
            aggregate=case["aggregate"],seed_map=run.map_metadata))
    metadata["reproduce"]=f"python tools/generate_{'environment' if mode=='environment' else 'sensor_state'}_video.py"
    output.with_suffix(".json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Saved {output} ({output.stat().st_size/1e6:.2f} MB)",flush=True)
    return metadata


def _inputs(directory, requested, *, sensor=False, regenerate=False, seed=1,
            allow_generate=True, prepare_all=False):
    directory=Path(directory).expanduser().resolve()
    required=list(SENSOR_CASES) if sensor else list(requested)
    if prepare_all:
        required=list(SENSOR_CASES)+["fog","rain"]
    seeds=(1,2,3) if sensor or prepare_all else (seed,)
    valid=(directory/"summary.json").exists()
    if valid:
        summary=_read(directory/"summary.json")
        valid=set(required).issubset({c["id"] for c in summary["cases"]})
        valid=valid and all((directory/"runs"/case/f"seed_{s}"/name).exists()
                           for case in required for s in seeds
                           for name in ("trajectory.json","trajectory.npz","mapping.json","mapping.npz"))
    if regenerate or not valid:
        if not allow_generate:
            raise FileNotFoundError("Saved video inputs are incomplete. Run python tools/generate_environment_video.py --prepare-only first.")
        from ..experiments.navigation_dependencies import run_study
        # Never overwrite frozen report data when missing raw archives in a clone.
        isolated=PROJECT/"data"/"video_inputs"
        if not directory.is_relative_to(isolated):
            directory=isolated/("all" if prepare_all else "sensor_states" if sensor else "environments")
        print(f"Generating missing video inputs in {directory}",flush=True)
        run_study(directory,seeds=seeds,case_ids=required)
        summary=_read(directory/"summary.json")
    return directory,summary


def video_main(mode, argv=None):
    parser=argparse.ArgumentParser(description="Render saved 3D simulation histories as a synchronized H.264 video")
    parser.add_argument("--data-dir",type=Path,default=PROJECT/"data"/"navigation_dependencies")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--seed",type=int,default=1)
    parser.add_argument("--fps",type=int,default=15,help="Encoded frame rate; numerical snapshots remain at their native 5 Hz")
    parser.add_argument("--width",type=int,default=1600);parser.add_argument("--height",type=int,default=900)
    parser.add_argument("--hold-seconds",type=float,default=3.)
    parser.add_argument("--ffmpeg",help="Optional ffmpeg executable path")
    parser.add_argument("--regenerate",action="store_true",help="Rerun only video-required studies into isolated data/video_inputs; preserves report data")
    parser.add_argument("--use-saved",action="store_true",help="Read-only rendering: fail if raw inputs are absent; never run numerical simulations")
    parser.add_argument("--prepare-only",action="store_true",help="Prepare all 12 video cases / three seeds without rendering; saves isolated inputs when missing")
    args=parser.parse_args(argv)
    if args.regenerate and args.use_saved:
        parser.error("--regenerate and --use-saved are mutually exclusive")
    if args.prepare_only:
        directory,_=_inputs(args.data_dir,(),prepare_all=True,regenerate=args.regenerate,allow_generate=not args.use_saved)
        print(f"Video inputs ready: {directory}",flush=True)
        return {"data_directory":str(directory)}
    if mode=="environment":
        ids=("baseline","fog","rain")
        directory,summary=_inputs(args.data_dir,ids,regenerate=args.regenerate,seed=args.seed,allow_generate=not args.use_saved)
        runs=[load_recording(directory,c,args.seed) for c in ids]
        labels=["NORMAL","FOG","RAIN"]
        params=["Clear air  |  extinction 0.0002 /m","Extinction 0.6 /m","Rainfall 50 mm/h  |  extinction 0.1048 /m"]
        title="3D LiDAR simulation in three environments"
        subtitle="One fixed obstacle scene and one sensor configuration. Synchronized navigation, raw scans and accumulated measured maps."
        selection=None;filename="environment_conditions.mp4"
    elif mode=="sensor":
        directory,summary=_inputs(args.data_dir,SENSOR_CASES,sensor=True,regenerate=args.regenerate,seed=args.seed,allow_generate=not args.use_saved)
        best,worst,ordered=select_sensor_extremes(summary)
        runs=[load_recording(directory,c["id"],args.seed) for c in (best,worst)]
        friendly={"pitch_up":"15° upward pitch","fov_side":"Side-facing 120° FOV"}
        labels=["BEST TESTED  |  "+friendly.get(best["id"],best["label"]),"WORST TESTED  |  "+friendly.get(worst["id"],worst["label"])]
        params=[]
        for run in runs:
            c,s=run.metadata["config"],run.metadata["sensor"];mount=c.get("sensor_mount",{})
            params.append(f"Mount {c['mount_height_m']:.1f} m  |  pitch {mount.get('pitch_deg',0):g}°  |  yaw {mount.get('yaw_deg',0):g}°  |  HFOV {s['hfov_end_deg']-s['hfov_start_deg']:g}°")
        title="Best and worst tested LiDAR configurations"
        subtitle="Best and worst among ten tested enabled configurations; task completion and safety rank before map coverage."
        selection={"eligible_cases":list(SENSOR_CASES),"ordering":RANKING_RULE,"ranked_case_ids":[c["id"] for c in ordered],
                   "exclusions":"sensor disabled, weather, target changes and added obstacles", "scope":"descriptive tested-case ranking, not a global sensor optimum",
                   "causal_limit":"the selected extremes differ in pitch, yaw and HFOV together; the video shows their combined configuration effect"}
        filename="sensor_state_best_vs_worst.mp4"
    else:
        raise ValueError(mode)
    return render_comparison(runs,labels,params,args.output or PROJECT/"simulalted_environments"/filename,
        title=title,subtitle=subtitle,summary=summary,summary_path=directory/"summary.json",mode=mode,selection=selection,
        fps=args.fps,width=args.width,height=args.height,hold_seconds=args.hold_seconds,ffmpeg=args.ffmpeg)
