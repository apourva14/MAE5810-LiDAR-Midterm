"""Application orchestration; numerical modules never depend on this layer."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

from . import api
from .paths import (DATA_DIR, OUTPUT_DIR, DEFAULT_SENSOR_CONFIG,
                    DEFAULT_NAVIGATION_CONFIG, DEFAULT_TRAJECTORY_CONFIG)


def load_navigation_config(path=DEFAULT_NAVIGATION_CONFIG):
    """Resolve configuration references against their file, independent of cwd."""
    path = Path(path).expanduser().resolve()
    nav = api.load_config_file(str(path))
    base = Path(nav["base_config"]).expanduser()
    if not base.is_absolute():
        base = path.parent / base
    nav["base_config"] = str(base.resolve())
    nav["_base_config_snapshot"] = api.load_config_file(nav["base_config"])
    return nav


def _scenarios(nav, selected=None):
    scenarios = [s for s in nav["scenarios"] if selected is None or s["name"] == selected]
    if not scenarios:
        raise ValueError(f"Unknown scenario {selected!r}; choose from " + ", ".join(s["name"] for s in nav["scenarios"]))
    return scenarios


def _visual_theme(path):
    from .visualization.theme import load_theme, apply_theme
    apply_theme(load_theme(path))


def scan(args):
    path = Path(args.config).expanduser().resolve()
    raw = api.apply_overrides(api.load_config_file(str(path)), args.set)
    cfg, pose, scene, sim, out = api.build_all(raw)
    destination = Path(args.data_dir) if args.data_dir else Path(out["directory"])
    if not args.data_dir and not destination.is_absolute():
        destination = path.parent / destination
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cloud = api.LidarSensor(cfg, pose, seed=sim["seed"]).scan(scene)
    for fmt in out["save"]:
        getattr(cloud, "save_" + fmt)(str(destination / ("pointcloud." + fmt)))
    snapshot = {
        "_source": str(path), "simulation": sim, "sensor": dataclasses.asdict(cfg),
        "pose": dataclasses.asdict(pose), "scene": raw.get("scene"), "output": out,
    }
    (destination / "config_used.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Saved {cloud.n:,} returns from {cloud.n_beams:,} beams to {destination} ({time.perf_counter()-start:.2f} s)")
    return 0


def render_scan(args):
    from .visualization.scan_figures import save_overview, save_report_figures
    from .visualization.scan_viewer import plot_interactive
    _visual_theme(args.theme)
    data = Path(args.data_dir).resolve()
    raw = api.load_config_file(str(Path(args.config) if args.config else data / "config_used.json"))
    cfg, pose, scene, _, out = api.build_all(raw)
    cloud = api.load_npz(data / "pointcloud.npz")
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    title = out.get("title") or "3D LiDAR saved scan"
    if args.format in ("all", "static"):
        save_overview(cloud, destination / "pointcloud.png", title,
                      scene=scene, cfg=cfg, sensor_position=pose.position)
        save_report_figures(cloud, scene, cfg, destination / "report_figures", sensor_position=pose.position)
    if args.format in ("all", "html"):
        plot_interactive(cloud, scene, cfg, pose.position,
                         str(destination / "pointcloud_interactive.html"), title)
    print(f"Rendered saved scan to {destination}")
    return 0


def navigate(args):
    from .navigation.engine import run
    from .navigation.storage import save_results
    nav = load_navigation_config(args.config)
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        nav["robot"]["max_steps"] = args.max_steps
    results = [run(nav, scenario) for scenario in _scenarios(nav, args.scenario)]
    save_results(results, nav, Path(args.data_dir).resolve())
    print(f"Saved navigation logs and replay data to {Path(args.data_dir).resolve()}")
    return 0


def _stored_navigation(args):
    from .navigation.storage import load_results
    # Prefer the configuration stored with a new run. Original CSV-only results
    # use the retained project configuration as their geometry reference.
    directory = Path(args.data_dir).resolve()
    nav = load_navigation_config(args.config) if args.config else None
    loaded = load_results(directory, nav=nav)
    # storage's return contract is (results, nav).
    results, nav = loaded
    if args.scenario:
        results = [r for r in results if r["name"] == args.scenario]
        if not results:
            raise ValueError(f"No saved results for scenario {args.scenario!r}")
    return results, nav


def render_navigation(args):
    from .visualization.navigation_figures import plot_trajectories, plot_clearance, plot_detection, animate
    from .navigation.storage import require_frames
    import matplotlib
    matplotlib.use("Agg")
    _visual_theme(args.theme)
    results, nav = _stored_navigation(args)
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plot_trajectories(results, nav, str(destination / "trajectories.png"))
    plot_clearance(results, nav, str(destination / "clearance.png"))
    plot_detection(results, nav, str(destination / "detection_range.png"))
    if args.gifs:
        for result in results:
            require_frames(result)
            animate(result, nav, str(destination / ("nav_" + result["name"] + ".gif")),
                    nav["output"].get("gif_every", 2), nav["output"].get("gif_fps", 8))
    print(f"Rendered navigation results to {destination}")
    return 0


def _display(args, nav, results=None):
    from .visualization.live import create_live_view
    from .navigation.engine import run
    import matplotlib.pyplot as plt
    _visual_theme(args.theme)
    if args.speed < 0:
        raise ValueError("--speed must be nonnegative")
    view = create_live_view(nav, speed=args.speed, backend=args.backend,
                            port=args.port, open_browser=not args.no_browser)
    names = [s["name"] for s in nav["scenarios"]]
    index = names.index(args.scenario) if args.scenario else 0
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plt.ion()
    try:
        while True:
            view.current, view.request = index, None
            reached_snapshot = False
            def display_step(state):
                nonlocal reached_snapshot
                keep_going = view.on_step(state)
                if args.snapshot is not None and state["k"] >= args.snapshot:
                    view.fig.savefig(destination / ("live_" + names[index] + ".png"), dpi=110)
                    reached_snapshot = True
                    return False
                return keep_going
            if results is None:
                run(nav, nav["scenarios"][index], on_step=display_step)
            else:
                from .visualization.live import iter_replay_states
                selected = next((r for r in results if r["name"] == names[index]), None)
                if selected is None:
                    raise ValueError(f"No saved replay for scenario {names[index]!r}")
                for state in iter_replay_states(selected):
                    if display_step(state) is False:
                        break
            if args.snapshot is not None:
                if not reached_snapshot:
                    raise ValueError("Requested snapshot is beyond the saved or simulated run")
                break
            if view.request == "quit":
                break
            while view.request is None:
                if not plt.fignum_exists(view.fig.number):
                    view.request = "quit"
                    break
                plt.pause(.1)
            if view.request == "quit":
                break
            index = int(view.request)
    finally:
        view.close()
        plt.close("all")
    return 0


def live(args):
    nav = load_navigation_config(args.config)
    _scenarios(nav, args.scenario)
    return _display(args, nav)


def replay(args):
    from .navigation.storage import require_history
    results, nav = _stored_navigation(args)
    for result in results:
        require_history(result)
    available = {result["name"] for result in results}
    nav = {**nav, "scenarios": [scenario for scenario in nav["scenarios"] if scenario["name"] in available]}
    if not args.scenario:
        args.scenario = results[0]["name"]
    return _display(args, nav, results=results)


def selftest(args):
    from .validation import selftest as run_tests
    return 0 if run_tests() else 1


def evaluate(args):
    from .experiments.runner import run_evaluation
    result = run_evaluation(args.data_dir, args.part)
    print(f"Saved controlled evaluation to {Path(args.data_dir).resolve()}")
    for name, study in result["studies"].items():
        print(f"  {name}: {study['elapsed_s']:.2f} s")
    return 0


def render_evaluation(args):
    from .visualization.evaluation import render_evaluation as render
    _visual_theme(args.theme)
    render(args.data_dir, args.output_dir)
    print(f"Rendered controlled evaluation to {Path(args.output_dir).resolve()}")
    return 0


def trajectory(args):
    from .trajectory import load_config, run_trajectory, save_result
    config = api.apply_overrides(load_config(args.config), args.set)
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        config["robot"]["max_steps"] = args.max_steps
    result = run_trajectory(config, args.scenario, seed=args.seed, record_history=True)
    save_result(result, Path(args.data_dir).resolve())
    print(f"Saved trajectory ({result['status']}) to {Path(args.data_dir).resolve()}")
    return 0


def evaluate_trajectory(args):
    from .trajectory import load_config
    from .experiments.trajectory import run_study
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain distinct integer seeds")
    result = run_study(Path(args.data_dir).resolve(), load_config(args.config), seeds=args.seeds)
    print(f"Saved trajectory factor study to {Path(args.data_dir).resolve()}")
    return 0


def render_trajectory(args):
    from .visualization.trajectory import render_trajectory as render
    _visual_theme(args.theme)
    render(args.data_dir, args.output_dir, run_name=args.run, gif=args.gifs,
           interactive=args.format in ("all", "html"))
    print(f"Rendered saved trajectories to {Path(args.output_dir).resolve()}")
    return 0


def map_environments(args):
    from .experiments.environments import run_comparison
    run_comparison(Path(args.data_dir).resolve(), args.config, seeds=args.seeds)
    print(f"Saved environment scans, growing maps and comparison to {Path(args.data_dir).resolve()}")
    return 0


def render_mapping(args):
    from .visualization.mapping import render_mapping as render
    _visual_theme(args.theme)
    render(args.data_dir, args.output_dir, interactive=args.format in ("all", "html"))
    print(f"Rendered saved environment maps to {Path(args.output_dir).resolve()}")
    return 0


def evaluate_navigation_dependencies(args):
    from .experiments.navigation_dependencies import run_study
    run_study(Path(args.data_dir).resolve(), args.config, seeds=args.seeds)
    print(f"Saved navigation dependency comparisons to {Path(args.data_dir).resolve()}")
    return 0


def render_navigation_dependencies(args):
    from .visualization.navigation_dependencies import render_navigation_dependencies as render
    _visual_theme(args.theme)
    render(args.data_dir, args.output_dir, interactive=args.format in ("all", "html"))
    if args.format in ("all", "static"):
        from .visualization.navigation_dependency_report import render_navigation_dependency_report
        render_navigation_dependency_report(args.data_dir, args.output_dir)
    print(f"Rendered saved navigation dependency comparisons to {Path(args.output_dir).resolve()}")
    return 0


def prepare_navigation_coverage(args):
    from .experiments.navigation_coverage import build_navigation_coverage
    result = build_navigation_coverage(args.source_dir, args.data_dir)
    print(f"Saved navigation FOV/LOS geometry to {result['path']}")
    return 0


def evaluate_navigation_threshold(args):
    from .experiments.navigation_threshold import run_study
    run_study(args.data_dir, source_dir=args.source_dir, trials_per_setting=args.trials)
    print(f"Saved navigation threshold trials to {Path(args.data_dir).resolve()}")
    return 0


def evaluate_navigation_robustness(args):
    from .experiments.navigation_robustness import run_study
    run_study(args.data_dir, config=args.config, seeds=args.seeds)
    print(f"Saved navigation robustness trials to {Path(args.data_dir).resolve()}")
    return 0


def diagnose_navigation_failures(args):
    from .experiments.navigation_robustness import summarize_failure_modes
    summarize_failure_modes(args.data_dir)
    print(f"Saved explanatory failure diagnostics to {Path(args.data_dir).resolve()}")
    return 0


def render_navigation_validation(args):
    from .visualization.navigation_validation import render_navigation_validation as render
    _visual_theme(args.theme)
    render(args.diagram, args.threshold_dir, args.robustness_dir, args.output_dir)
    print(f"Rendered saved navigation validation to {Path(args.output_dir).resolve()}")
    return 0


def init_config(args):
    path = Path(args.output).expanduser().resolve()
    if path.exists() and not args.force:
        raise ValueError(f"{path} already exists; use --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(api.CONFIG_TEMPLATE, encoding="utf-8")
    print(f"Wrote {path}")
    return 0


def parser():
    ap = argparse.ArgumentParser(description="LiDAR simulation and independent saved-data visualization")
    sub = ap.add_subparsers(dest="command", required=True)
    def command(name, help, fn):
        item = sub.add_parser(name, help=help)
        item.set_defaults(func=fn)
        return item
    def theme(p):
        p.add_argument("--theme", help="Partial visualization theme JSON; affects presentation only")
    def stored(p):
        p.add_argument("--data-dir", default=str(DATA_DIR / "navigation"))
        p.add_argument("--config", help="Configuration for historical CSV-only results; new runs use their saved input snapshot")
        p.add_argument("--scenario")
    def display(p):
        p.add_argument("--backend", choices=["matplotlib", "browser"], default="matplotlib")
        p.add_argument("--speed", type=float, default=1.0)
        p.add_argument("--port", type=int, default=8765)
        p.add_argument("--no-browser", action="store_true")
        p.add_argument("--snapshot", type=int, help="Save step N and exit")
        p.add_argument("--output-dir", default=str(OUTPUT_DIR / "navigation"))
        theme(p)
    p = command("scan", "Simulate sensor data without loading visualization libraries", scan)
    p.add_argument("--config", default=str(DEFAULT_SENSOR_CONFIG))
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--data-dir", help="Override the configuration's numerical output directory")
    p = command("render-scan", "Render an existing NPZ scan without simulating", render_scan)
    p.add_argument("--data-dir", default=str(DATA_DIR / "scan"))
    p.add_argument("--config", help="Sensor config override; defaults to saved config_used.json")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR / "scan"))
    p.add_argument("--format", choices=["all", "static", "html"], default="all")
    theme(p)
    p = command("navigate", "Simulate robot navigation and save numerical replay data", navigate)
    p.add_argument("--config", default=str(DEFAULT_NAVIGATION_CONFIG))
    p.add_argument("--scenario")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--data-dir", default=str(DATA_DIR / "navigation"))
    p = command("render-navigation", "Render existing navigation logs and optional stored animations", render_navigation)
    stored(p)
    p.add_argument("--output-dir", default=str(OUTPUT_DIR / "navigation"))
    p.add_argument("--gifs", action="store_true", help="Requires stored frames; never reruns simulation")
    theme(p)
    p = command("live", "Display a navigation simulation as it runs", live)
    p.add_argument("--config", default=str(DEFAULT_NAVIGATION_CONFIG))
    p.add_argument("--scenario")
    display(p)
    p = command("replay", "Replay stored numerical frames with the current visualization style", replay)
    stored(p); display(p)
    command("selftest", "Run the analytical sensor checks", selftest)
    p = command("evaluate", "Run controlled geometry, target, atmosphere and uncertainty studies", evaluate)
    p.add_argument("--part", choices=["all", "geometry", "response"], default="all")
    p.add_argument("--data-dir", default=str(DATA_DIR / "evaluation"))
    p = command("render-evaluation", "Plot existing controlled-study results without simulating", render_evaluation)
    p.add_argument("--data-dir", default=str(DATA_DIR / "evaluation"))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR / "evaluation"))
    theme(p)
    p = command("trajectory", "Generate a reference route and simulate LiDAR-guided tracking", trajectory)
    p.add_argument("--config", default=str(DEFAULT_TRAJECTORY_CONFIG))
    p.add_argument("--scenario", default="baseline")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--data-dir", default=str(DATA_DIR / "trajectory" / "demo"))
    p = command("evaluate-trajectory", "Vary one trajectory or sensing factor at a time", evaluate_trajectory)
    p.add_argument("--config", default=str(DEFAULT_TRAJECTORY_CONFIG))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--data-dir", default=str(DATA_DIR / "trajectory"))
    p = command("render-trajectory", "Render stored routes, measurements and factor comparisons", render_trajectory)
    p.add_argument("--data-dir", default=str(DATA_DIR / "trajectory"))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR / "trajectory"))
    p.add_argument("--run", help="Stored study run or case name; defaults to a baseline run")
    p.add_argument("--format", choices=["all", "static", "html"], default="all")
    p.add_argument("--gifs", action="store_true", help="Animate saved motion and LiDAR obstacle returns")
    theme(p)
    p = command("map-environments", "Compare moving 3D LiDAR maps in open, cluttered and narrow scenes", map_environments)
    p.add_argument("--config", default=str(DEFAULT_TRAJECTORY_CONFIG))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--data-dir", default=str(DATA_DIR / "mapping"))
    p = command("render-mapping", "Render saved 3D maps and environment comparisons", render_mapping)
    p.add_argument("--data-dir", default=str(DATA_DIR / "mapping"))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR / "mapping"))
    p.add_argument("--format", choices=["all", "static", "html"], default="all")
    theme(p)
    p = command("evaluate-navigation-dependencies", "Compare LiDAR dependencies using matched poses and closed-loop navigation", evaluate_navigation_dependencies)
    p.add_argument("--config", default=str(DEFAULT_TRAJECTORY_CONFIG))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--data-dir", default=str(DATA_DIR / "navigation_dependencies"))
    p = command("render-navigation-dependencies", "Render saved navigation dependency measurements and moving 3D maps", render_navigation_dependencies)
    p.add_argument("--data-dir", default=str(DATA_DIR / "navigation_dependencies"))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR / "navigation_dependencies"))
    p.add_argument("--format", choices=["all", "static", "html"], default="all")
    theme(p)
    p = command("prepare-navigation-coverage", "Prepare FOV/LOS ray geometry at a saved navigation pose", prepare_navigation_coverage)
    p.add_argument("--source-dir", default=str(DATA_DIR / "navigation_dependencies"), help="Existing navigation dependency study")
    p.add_argument("--data-dir", default=str(DATA_DIR / "navigation_validation"), help="Destination for prepared geometry")
    p = command("evaluate-navigation-threshold", "Run independent near-threshold scan trials at frozen navigation viewpoints", evaluate_navigation_threshold)
    p.add_argument("--source-dir", default=str(DATA_DIR / "navigation_dependencies"), help="Existing navigation dependency study")
    p.add_argument("--data-dir", default=str(DATA_DIR / "navigation_threshold"))
    p.add_argument("--trials", type=int, default=1000, help="Independent trials per viewpoint and SNR setting (default: 1000)")
    p = command("evaluate-navigation-robustness", "Repeat closed-loop sensing conditions across three fixed layouts", evaluate_navigation_robustness)
    p.add_argument("--config", default=str(DEFAULT_TRAJECTORY_CONFIG))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1001, 1021)), help="Distinct noise-seed labels (default: 1001 through 1020)")
    p.add_argument("--data-dir", default=str(DATA_DIR / "navigation_robustness"))
    p = command("diagnose-navigation-failures", "Explain selected failures from the completed default robustness study", diagnose_navigation_failures)
    p.add_argument("--data-dir", default=str(DATA_DIR / "navigation_robustness"))
    p = command("render-navigation-validation", "Render saved FOV/LOS, threshold and robustness evidence without simulating", render_navigation_validation)
    p.add_argument("--diagram", default=str(DATA_DIR / "navigation_validation" / "fov_los.json"))
    p.add_argument("--threshold-dir", default=str(DATA_DIR / "navigation_threshold"))
    p.add_argument("--robustness-dir", default=str(DATA_DIR / "navigation_robustness"))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR / "navigation_validation"))
    theme(p)
    p = command("init-config", "Write a documented sensor configuration", init_config)
    p.add_argument("--output", default=str(DEFAULT_SENSOR_CONFIG))
    p.add_argument("--force", action="store_true")
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (api.ConfigError, ValueError, FileNotFoundError, KeyError) as exc:
        ap.exit(2, f"{exc}\n")
    except ImportError as exc:
        ap.exit(2, f"{exc}. Install the optional libraries with: python -m pip install -e '.[visualization]'\n")


if __name__ == "__main__":
    raise SystemExit(main())
