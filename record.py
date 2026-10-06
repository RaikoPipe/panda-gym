"""Record teaching videos from a training run (see train.py).

    python record.py runs/Push_crossq_s0 PandaPush-v3
    python record.py runs/ReachAO_crossq_s0 PandaReachAO-v3 --scenario library2 --max-episode-steps 150

Writes to videos/: <name>_training.mp4 (one clip per checkpoint, showing learning progress) and
<name>_deploy.mp4 (the best model on fresh episodes).
"""
import argparse
import re
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sb3_contrib import TQC, CrossQ

import panda_gym  # noqa: F401

FPS = 25


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def load(path):
    for cls in (CrossQ, TQC):
        try:
            # playback only: skip allocating the (HER) replay buffer
            return cls.load(path, device="cpu", custom_objects={"replay_buffer_class": None, "replay_buffer_kwargs": {}, "buffer_size": 1})
        except (ValueError, KeyError, RuntimeError, TypeError):
            continue
    raise ValueError(f"cannot load {path}")


def overlay(frame, title, subtitle, status=None):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    h = img.height
    draw.rounded_rectangle((16, 16, 16 + int(h * 0.75), 16 + int(h * 0.13)), radius=12, fill=(20, 24, 32, 170))
    draw.text((32, 24), title, font=font(int(h * 0.045)), fill="white")
    draw.text((32, 24 + int(h * 0.055)), subtitle, font=font(int(h * 0.032)), fill=(200, 210, 225))
    if status:
        text, color = status
        f = font(int(h * 0.05))
        w = draw.textlength(text, font=f)
        draw.rounded_rectangle((img.width - w - 48, 16, img.width - 16, 16 + int(h * 0.08)), radius=12, fill=color)
        draw.text((img.width - w - 32, 22), text, font=f, fill="white")
    return np.asarray(img)


def episode(model, env, seed, title, subtitle, orbit):
    """Roll out one deterministic episode and return annotated frames."""
    obs, _ = env.reset(seed=seed)
    frames, done, info = [], False, {}
    while not done:
        frames.append(overlay(env.render(), title, subtitle))
        env.unwrapped.render_kwargs["yaw"] += orbit
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    if info.get("is_success"):
        status = ("SUCCESS", (30, 160, 80, 230))
    elif info.get("is_collided"):
        status = ("COLLISION", (200, 50, 50, 230))
    else:
        status = ("TIMEOUT", (120, 120, 120, 230))
    frames += [overlay(env.render(), title, subtitle, status)] * (FPS // 2)  # hold the outcome for half a second
    return frames, status[0] == "SUCCESS"


def main(args):
    run = Path(args.run)
    name = run.name + (f"_{args.scenario}" if args.scenario else "")
    kwargs = {"scenario": args.scenario} if args.scenario else {}
    if args.max_episode_steps:
        kwargs["max_episode_steps"] = args.max_episode_steps
    env = gym.make(args.env, render_mode="rgb_array", **kwargs)
    env.unwrapped.render_kwargs.update(width=args.width, height=args.height)
    task = args.env.removesuffix("-v3") + (f" · {args.scenario}" if args.scenario else "")
    out = Path("videos")
    out.mkdir(exist_ok=True)

    def steps(p):
        return int(re.search(r"_(\d+)_steps", p.name).group(1))

    checkpoints = sorted(run.glob(f"checkpoints/*{args.checkpoint_filter}*_steps.zip"), key=steps)
    if checkpoints:
        with imageio.get_writer(out / f"{name}_training.mp4", fps=FPS, quality=8) as writer:
            for ckpt in checkpoints:
                model = load(ckpt)
                for i in range(args.episodes_per_checkpoint):
                    frames, _ = episode(model, env, 1000 + i, task, f"training step {steps(ckpt):,}", args.orbit)
                    for f in frames:
                        writer.append_data(f)
                print(f"{ckpt.name} done")

    best = run / args.model
    model = load(best if best.exists() else sorted(run.glob("best*/best_model.zip"))[-1])
    successes = 0
    with imageio.get_writer(out / f"{name}_deploy.mp4", fps=FPS, quality=8) as writer:
        for i in range(args.episodes):
            frames, ok = episode(model, env, 2000 + i, task, f"trained agent · episode {i + 1}/{args.episodes}", args.orbit)
            successes += ok
            for f in frames:
                writer.append_data(f)
    print(f"{name}: deployment success {successes}/{args.episodes}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("env")
    parser.add_argument("--scenario")
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--model", default="final.zip", help="model for the deployment video (falls back to best/)")
    parser.add_argument("--checkpoint-filter", default="", help="e.g. a curriculum stage name")
    parser.add_argument("--episodes-per-checkpoint", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--orbit", type=float, default=0.15, help="camera yaw change per frame (degrees)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    main(parser.parse_args())
