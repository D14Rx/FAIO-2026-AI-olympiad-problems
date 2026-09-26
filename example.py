"""
Minimal example: run one episode in the RobotWalker environment and visualize it.

    python example_faio.py                                # random actions
    python example_faio.py --submission submission.npz    # your trained policy
    python example_faio.py --seed 3 --out my_run          # choose episode, output name

Creates <out>.png and <out>.gif.  Needs matplotlib:  pip install matplotlib
Put faio_logo.png next to this file to show the logo (otherwise a drawn mark is used).
"""

import argparse
import os
import tempfile
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from environment import RobotWalker, MAX_STEPS, OBS_DIM, ACT_DIM
from policy import act, load_submission

HERE = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(HERE, "faio_logo.png")
DT = 0.008  # seconds per environment step

# ---- faio.kz palette --------------------------------------------------------
BLUE, NAVY, BLUE_HI = "#001DDD", "#01139B", "#2C40C7"
GOLD, GOLD_DARK = "#E0A32C", "#A8731A"
PAPER, LAND, HAIR = "#F1F4F5", "#D5DDE0", "#C4CFD3"
INK, MUTED = "#11242E", "#5D7580"
SKY, SKY_LIGHT = "#C3CDFF", "#F4F6FF"

RENDER_W, RENDER_H = 960, 540   # MuJoCo render size
OUT_W, OUT_H = 640, 360         # GIF size


def rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def mj(h, a=None):
    """'#rrggbb' -> 'r g b [a]' in 0..1 for MuJoCo XML."""
    s = " ".join(f"{c / 255:.3f}" for c in rgb(h))
    return s if a is None else f"{s} {a}"


# ---- 1. Themed MuJoCo model (colours only) ------------------------------------
def make_floor_tile(path, n=256):
    """One floor tile: light ground with a diamond lattice and FAIO-blue diamonds."""
    img = Image.new("RGB", (n, n), rgb(LAND))
    d = ImageDraw.Draw(img)
    d.line([(0, n // 2), (n // 2, 0), (n, n // 2), (n // 2, n), (0, n // 2)], fill=rgb(MUTED), width=3)
    c, s = n // 2, n // 9
    d.polygon([(c, c - s), (c + s, c), (c, c + s), (c - s, c)], fill=rgb(BLUE))
    s2 = n // 20
    for x, y in [(0, 0), (n, 0), (0, n), (n, n)]:  # quarter diamonds -> full diamond across 4 tiles
        d.polygon([(x, y - s2), (x + s2, y), (x, y + s2), (x - s2, y)], fill=rgb(NAVY))
    img.save(path)


def build_theme_xml(out_dir):
    """Copy Gymnasium's walker2d_v5.xml and change only visual things: textures,
    materials, rgba colours, lights and render size. Joints, geoms, masses,
    friction and actuators are untouched, so the physics is identical."""
    import gymnasium.envs.mujoco as mj_envs
    src = os.path.join(os.path.dirname(mj_envs.__file__), "assets", "walker2d_v5.xml")
    tree = ET.parse(src)
    root = tree.getroot()

    asset = root.find("asset")
    for child in list(asset):
        asset.remove(child)
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1=mj(SKY), rgb2=mj(SKY_LIGHT), width="256", height="256")
    ET.SubElement(asset, "texture", name="texplane", type="2d", file="floor.png")
    ET.SubElement(asset, "material", name="MatPlane", texture="texplane", texuniform="true",
                  texrepeat="0.6 0.6", reflectance="0", shininess="0", specular="0.05")
    make_floor_tile(os.path.join(out_dir, "floor.png"))

    colours = {
        "torso_geom": BLUE,
        "thigh_geom": BLUE_HI, "leg_geom": BLUE_HI, "foot_geom": GOLD,
        "thigh_left_geom": NAVY, "leg_left_geom": NAVY, "foot_left_geom": GOLD,
    }
    for g in root.iter("geom"):
        name = g.get("name")
        if name in colours:
            g.set("rgba", mj(colours[name], 1))
        elif name == "floor":
            g.set("rgba", "1 1 1 1")

    for cam in root.iter("camera"):  # the tracking camera: a bit further back and lower
        if cam.get("name") == "track":
            cam.set("pos", "0 -3.6 -0.45")
            cam.set("xyaxes", "1 0 0 0 0 1")
    for light in root.iter("light"):  # softer light so the light floor keeps its colour
        light.set("diffuse", ".7 .7 .7")
        light.set("specular", "0 0 0")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "headlight", ambient=".25 .25 .25", diffuse=".35 .35 .35", specular="0 0 0")
    ET.SubElement(vis, "global", offwidth=str(RENDER_W), offheight=str(RENDER_H))
    ET.SubElement(vis, "quality", shadowsize="4096")

    path = os.path.join(out_dir, "walker2d_faio.xml")
    tree.write(path)
    return path


class FAIORobotWalker(RobotWalker):
    """RobotWalker with the FAIO paint job. Same observations, actions, rewards, termination."""

    def __init__(self, xml_path):
        env = gym.make(
            "Walker2d-v5", max_episode_steps=MAX_STEPS, render_mode="rgb_array",
            xml_file=xml_path, width=RENDER_W, height=RENDER_H, camera_name="track",
        )
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)
        self.v_target = None


# ---- 2. Overlay: header, ornament, speed gauge --------------------------------
def load_font(size, bold=False):
    weight = "bold" if bold else "normal"
    for family in ("Noto Sans", "DejaVu Sans"):
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=family, weight=weight),
                                         fallback_to_default=False)
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def load_logo(height):
    if os.path.exists(LOGO_PATH):
        logo = Image.open(LOGO_PATH).convert("RGBA")
        return logo.resize((round(logo.width * height / logo.height), height), Image.LANCZOS)
    # fallback: the FAIO diamond mark drawn by hand
    logo = Image.new("RGBA", (height, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(logo)
    c, s, t = height / 2, height * 0.30, height * 0.11
    d.polygon([(c, c - s), (c + s, c), (c, c + s), (c - s, c)], fill=rgb(BLUE))
    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        x, y = c + dx * (s + t + 2), c + dy * (s + t + 2)
        d.polygon([(x, y - t), (x + t, y), (x, y + t), (x - t, y)], fill=rgb(NAVY))
    return logo


def ornament_strip(d, y, width, h=16, period=32, colour=BLUE, bg=SKY):
    """Repeating 'qoshqar muiz' (ram horn) motif, a classic Kazakh ornament."""
    d.rectangle([0, y, width, y + h], fill=rgb(bg))
    r = (h - 4) // 2
    yc = y + 2 + r
    for x in range(period // 2, width + period, period):
        d.line([(x, y + h - 1), (x, yc)], fill=rgb(colour), width=2)
        d.arc([x - 2 * r, yc - r, x, yc + r], start=120, end=360, fill=rgb(colour), width=2)
        d.arc([x, yc - r, x + 2 * r, yc + r], start=180, end=420, fill=rgb(colour), width=2)


class Overlay:
    TOP = 56          # header height
    STRIP = 14        # ornament strip height
    BOT = 46          # HUD height (one compact row so the feet stay visible)
    V_MAX = 6.0       # gauge range

    def __init__(self):
        self.f_title = load_font(24, bold=True)
        self.f_small = load_font(14)
        self.f_hud = load_font(15, bold=True)
        self.f_tiny = load_font(11)
        self.f_badge = load_font(28, bold=True)
        self.logo = load_logo(self.TOP - 16)

    def draw(self, frame, v_target, v, step, ret, badge=None):
        img = Image.fromarray(frame).resize((OUT_W, OUT_H), Image.LANCZOS)
        d = ImageDraw.Draw(img)
        W, H, TOP = OUT_W, OUT_H, self.TOP
        BOT = H - self.BOT

        # header
        d.rectangle([0, 0, W, TOP], fill=rgb(PAPER))
        img.paste(self.logo, (14, (TOP - self.logo.height) // 2), self.logo)
        x0 = 14 + self.logo.width + 14
        d.text((x0, TOP / 2), "RobotWalker", font=self.f_title, fill=rgb(INK), anchor="lm")
        d.text((W - 14, TOP / 2 - 9), "FAIO 2026", font=self.f_small, fill=rgb(NAVY), anchor="rm")
        d.text((W - 14, TOP / 2 + 9), "faio.kz", font=self.f_small, fill=rgb(MUTED), anchor="rm")
        ornament_strip(d, TOP, W, h=self.STRIP)

        # HUD: one row -> target | speed | gauge | time and return
        cy = BOT + self.BOT // 2
        d.rectangle([0, BOT, W, H], fill=rgb(PAPER))
        d.line([0, BOT, W, BOT], fill=rgb(HAIR), width=2)
        d.text((12, cy), f"target {v_target:.2f} m/s", font=self.f_hud, fill=rgb(GOLD_DARK), anchor="lm")
        d.text((150, cy), f"speed {v:5.2f} m/s", font=self.f_hud, fill=rgb(BLUE), anchor="lm")

        bx0, bx1, by = 290, 420, cy + 3
        d.rounded_rectangle([bx0, by - 5, bx1, by + 5], radius=5, fill=rgb(LAND))
        xv = bx0 + (bx1 - bx0) * min(max(v, 0.0), self.V_MAX) / self.V_MAX
        if xv > bx0 + 10:
            d.rounded_rectangle([bx0, by - 5, xv, by + 5], radius=5, fill=rgb(BLUE))
        xt = bx0 + (bx1 - bx0) * v_target / self.V_MAX
        d.polygon([(xt, by - 7), (xt - 6, by - 16), (xt + 6, by - 16)], fill=rgb(GOLD))
        d.line([(xt, by - 5), (xt, by + 5)], fill=rgb(GOLD), width=2)
        d.text((bx1 + 6, by), f"{self.V_MAX:.0f} m/s", font=self.f_tiny, fill=rgb(MUTED), anchor="lm")

        d.text((W - 12, cy - 9), f"t = {step * DT:5.2f} s   step {step:4d}", font=self.f_tiny,
               fill=rgb(MUTED), anchor="rm")
        d.text((W - 12, cy + 9), f"return {ret:7.1f}", font=self.f_hud, fill=rgb(INK), anchor="rm")

        if badge:
            colour = BLUE if badge == "FINISH" else GOLD
            tw = d.textlength(badge, font=self.f_badge)
            cx, cy = W / 2, TOP + self.STRIP + 34
            d.rounded_rectangle([cx - tw / 2 - 18, cy - 22, cx + tw / 2 + 18, cy + 22], radius=10, fill=rgb(colour))
            d.text((cx, cy), badge, font=self.f_badge, fill="white", anchor="mm")
        return img


# ---- 3. Main -------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submission", default=None, help="submission.npz; random actions if omitted")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="faio_run")
    p.add_argument("--every", type=int, default=8, help="keep every n-th frame in the GIF")
    args = p.parse_args()

    w = load_submission(args.submission) if args.submission else None

    with tempfile.TemporaryDirectory() as tmp:
        env = FAIORobotWalker(build_theme_xml(tmp))
        obs, info = env.reset(seed=args.seed)
        v_target = info["v_target"]
        speeds, rewards, frames = [], [], []
        done = False
        while not done:
            action = act(obs, w) if w is not None else env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            speeds.append(info["x_velocity"])
            rewards.append(reward)
            frames.append(env.render())
            done = terminated or truncated
        env.close()

    steps = len(rewards)
    fell = steps < MAX_STEPS
    print(f"seed {args.seed}: v_target {v_target:.2f} m/s, {steps} steps, return {sum(rewards):.2f}, "
          f"{'fell' if fell else 'survived'}")

    # plots in FAIO colours
    plt.rcParams.update({"figure.facecolor": PAPER, "axes.facecolor": "white", "axes.edgecolor": HAIR,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "grid.color": HAIR, "axes.grid": True})
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [2, 2, 1.6]})
    fig.suptitle(f"RobotWalker · FAIO 2026 · seed {args.seed}", fontsize=15, fontweight="bold", color=NAVY)

    axes[0].plot(speeds, color=BLUE, label="robot speed")
    axes[0].axhline(v_target, color=GOLD, linestyle="--", label=f"target {v_target:.2f} m/s")
    axes[0].set_ylabel("speed, m/s"); axes[0].legend(); axes[0].set_title("Speed vs target")

    axes[1].plot(rewards, color=BLUE_HI)
    axes[1].set_ylabel("reward"); axes[1].set_xlabel("step")
    axes[1].set_title(f"Reward per step (return {sum(rewards):.1f})")

    picks = np.linspace(0, steps - 1, 6, dtype=int)
    strip = np.concatenate([frames[i] for i in picks], axis=1)
    axes[2].imshow(strip); axes[2].axis("off"); axes[2].grid(False)
    axes[2].set_title("Frames at steps " + ", ".join(str(i) for i in picks))

    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=110)

    # GIF with header, ornament and gauge
    overlay = Overlay()
    idx = list(range(0, steps, args.every))
    if idx[-1] != steps - 1:
        idx.append(steps - 1)
    cum = np.cumsum(rewards)
    imgs, durations = [], []
    for i in idx:
        badge = ("FELL" if fell else "FINISH") if i == steps - 1 else None
        imgs.append(overlay.draw(frames[i], v_target, speeds[i], i + 1, cum[i], badge))
        durations.append(int(args.every * DT * 1000))
    durations[-1] = 1500  # hold the last frame
    # one shared 128-colour palette for all frames keeps the file about half the size
    palette = imgs[len(imgs) // 2].quantize(colors=128, method=Image.MEDIANCUT)
    imgs = [im.quantize(palette=palette, dither=Image.NONE) for im in imgs]
    imgs[0].save(f"{args.out}.gif", save_all=True, append_images=imgs[1:], duration=durations,
                 loop=0, optimize=True)
    print(f"saved {args.out}.png and {args.out}.gif")


if __name__ == "__main__":
    main()
