#!/usr/bin/env python3
"""Join the saved simulation videos into one narrated four-scenario video.

Run: python tools/generate_all_scenarios_video.py [--ffmpeg PATH]
Inputs are the three saved videos in simulalted_environments/ and the saved
target-characteristics figure. No sensor or navigation simulation is rerun;
title cards only restate numbers already stored in data/ and outputs/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
VIDEOS = PROJECT / "simulalted_environments"
WIDTH, HEIGHT, FPS = 1920, 1080, 15
BACKGROUND, INK, MUTED = "#F7F9FC", "#19394F", "#40576A"
ACCENTS = ("#2875A2", "#8B68B0", "#38988D", "#AF673E")
FONTS = ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

SCENARIOS = [
    dict(title="Scenario 1 · Environmental conditions",
         lead="Same scene, robot, sensor, controller and planned route. Only atmospheric extinction κ changes.",
         bullets=["Normal air: κ = 0.0002 /m",
                  "Rain 50 mm/h: κ = 0.1048 /m",
                  "Fog: κ = 0.6 /m"],
         watch=["Points per scan fall 2,198 → 2,011 → 1,742 (normal → rain → fog).",
                "Observed map cells fall 73.7% → 70.3% → 47.1%.",
                "All nine closed-loop runs still reach the goal without collision."],
         video="environment_conditions.mp4"),
    dict(title="Scenario 2 · Sensor configuration: best vs worst",
         lead="Matched-pose replay: identical poses, mount angles, FOV, receiver noise and weather.",
         bullets=["Best: 360 azimuths × 8 channels, target diffuse reflectance 0.85",
                  "Worst: 90 azimuths × 8 channels, target reflectance 0.00005 (virtual stress setting)"],
         watch=["Sparse sampling returns about one quarter of the points.",
                "The weak target is detected in about 75% of eligible echoes.",
                "Its detected range RMS is 22 mm instead of 10 mm."],
         video="sensor_state_best_vs_worst.mp4"),
    dict(title="Scenario 3 · Sampling density × target reflectance",
         lead="All four combinations at the same poses: rows change sampling, columns change reflectance.",
         bullets=["Sampling: 360 or 90 azimuth samples per revolution",
                  "Target diffuse reflectance: 0.85 or 0.00005"],
         watch=["Sampling sets how many points: 549 vs 2,198 per scan.",
                "Reflectance sets detectability: 100% vs 74.5% of eligible target echoes.",
                "Reflectance sets precision: 10.0 vs 22.3 mm range RMS; sampling does not."],
         video="sampling_reflectance_factorial.mp4"),
    dict(title="Scenario 4 · Target state and features",
         lead="Nine target settings replayed over the same 271 baseline poses.",
         bullets=["Reflectance 0.02 / 0.30 / 0.85  ·  yaw 0° / 45° / 75°",
                  "Height 0.3 / 1.2 / 2.0 m  ·  centre x = 9 / 11 / 13 m"],
         watch=["Reflectance scales return power linearly (0.067×, 1×, 2.83×) with the same rays hit.",
                "Height, yaw and position change which rays hit, the incidence angle and the echo count.",
                "Every setting stays fully detected with about 10 mm range RMS."],
         figure="outputs/all_case_report/target_characteristics.png"),
]


def font(size, bold=False):
    for path in FONTS[1::2] if bold else FONTS[::2]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def canvas():
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    return image, ImageDraw.Draw(image)


def title_card(path, accent, kicker, title, lead, sections):
    image, draw = canvas()
    draw.rectangle([0, 0, 24, HEIGHT], fill=accent)
    draw.text((140, 150), kicker, font=font(30, True), fill=accent)
    draw.text((140, 200), title, font=font(62, True), fill=INK)
    draw.text((140, 300), lead, font=font(32), fill=MUTED)
    y = 400
    for heading, lines in sections:
        draw.text((140, y), heading, font=font(30, True), fill=INK)
        y += 56
        for line in lines:
            draw.text((180, y), "•  " + line, font=font(32), fill=MUTED)
            y += 54
        y += 34
    image.save(path)


def figure_card(path, accent, title, figure):
    image, draw = canvas()
    draw.rectangle([0, 0, WIDTH, 16], fill=accent)
    draw.text((80, 50), title, font=font(44, True), fill=INK)
    draw.text((80, 112), "Saved results from the matched-pose replay (outputs/all_case_report/target_metrics.csv)",
              font=font(26), fill=MUTED)
    plot = Image.open(PROJECT / figure).convert("RGB")
    plot.thumbnail((WIDTH - 160, HEIGHT - 230))
    image.paste(plot, ((WIDTH - plot.width) // 2, 190 + (HEIGHT - 230 - plot.height) // 2))
    image.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ffmpeg", help="Optional ffmpeg executable path")
    parser.add_argument("--output", type=Path, default=VIDEOS / "lidar_all_scenarios.mp4")
    args = parser.parse_args(argv)
    executable = args.ffmpeg or shutil.which("ffmpeg")
    if not executable:
        raise SystemExit("ffmpeg is required (macOS: brew install ffmpeg)")
    work = PROJECT / "build" / "all_scenarios"
    work.mkdir(parents=True, exist_ok=True)

    # (image or video path, seconds for images; None for videos)
    clips = []
    intro = work / "00_intro.png"
    title_card(intro, INK, "MAE 5810 Robot Perception · Midterm Project · Apurva Singh",
               "Physics-based 3D LiDAR simulation",
               "How measurements and field of view depend on sensor state, environment and target.",
               [("Four scenarios", [s["title"] for s in SCENARIOS]),
                ("Every scene", ["8-channel spinning pulsed LiDAR, 0.6 m mount, fixed obstacle course, 5 Hz scans",
                                 "Green: nominal FOV  ·  orange: latest scan  ·  colour: accumulated map  ·  pink: target returns"])])
    clips.append((intro, 7))
    for i, scenario in enumerate(SCENARIOS):
        card = work / f"{i + 1:02d}_card.png"
        title_card(card, ACCENTS[i], f"Scenario {i + 1} of 4", scenario["title"].split(" · ", 1)[1],
                   scenario["lead"], [("Setup", scenario["bullets"]), ("What to look for", scenario["watch"])])
        clips.append((card, 8))
        if "video" in scenario:
            clips.append((VIDEOS / scenario["video"], None))
        else:
            still = work / f"{i + 1:02d}_figure.png"
            figure_card(still, ACCENTS[i], scenario["title"], scenario["figure"])
            clips.append((still, 14))
    outro = work / "99_outro.png"
    title_card(outro, INK, "Summary", "What the four scenarios show",
               "Each dependency enters through a different term of the sensor model.",
               [("Dependencies", ["Environment: extinction κ thins returns and coverage, not range accuracy",
                                  "Sensor configuration: sampling density sets point count, not precision",
                                  "Target features: reflectance sets power, detectability and precision",
                                  "Target state: position, yaw and height set geometry, FOV and LOS"]),
                ("Scope", ["Weather is attenuation only (no backscatter); materials are illustrative",
                           "Code and data: github.com/apourva14/MAE5810-LiDAR-Midterm"])])
    clips.append((outro, 8))

    command = [executable, "-hide_banner", "-loglevel", "error", "-y"]
    for path, seconds in clips:
        command += ["-loop", "1", "-framerate", str(FPS), "-t", str(seconds), "-i", str(path)] if seconds else ["-i", str(path)]
    scale = (f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
             f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0xF7F9FC,fps={FPS},format=yuv420p,setsar=1")
    graph = ";".join(f"[{i}:v]{scale}[v{i}]" for i in range(len(clips)))
    graph += ";" + "".join(f"[v{i}]" for i in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0[out]"
    command += ["-filter_complex", graph, "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "slow",
                "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
    subprocess.run(command, check=True)
    manifest = dict(video=args.output.name, frame_size=[WIDTH, HEIGHT], fps=FPS,
                    segments=[dict(source=str(Path(p).relative_to(PROJECT)), seconds=s) for p, s in clips],
                    note="Concatenation of saved media; no simulation is rerun.",
                    reproduce="python tools/generate_all_scenarios_video.py")
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output} ({args.output.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
