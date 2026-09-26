"""
train.py - обучение PPO для RobotWalker (FAIO 2026) и сохранение submission.npz.

Положи рядом с environment.py и policy.py, затем:
    pip install "gymnasium[mujoco]" stable-baselines3
    python train.py                       # 10 млн шагов
    python train.py --steps 30000000      # дольше = лучше
    python train.py --resume              # продолжить с чекпоинта

Лучший по оценке файл всегда лежит в submission.npz.
"""

import argparse
import os
import shutil

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from environment import RobotWalker, MAX_STEPS
from policy import act, load_submission, save_submission

MAX_RETURN = 1.05 * MAX_STEPS  # 1312.5
GAMMA = 0.995
BATCH_TOTAL = 16384  # шагов за один сбор данных (по всем средам)


def export_weights(model, venv):
    """Достаёт веса сети 18->64->64->6 и статистику нормализации в формате policy.py."""
    p = model.policy
    net = p.mlp_extractor.policy_net  # [Linear, Tanh, Linear, Tanh]
    g = lambda t: t.detach().cpu().numpy().astype(np.float64)
    return dict(
        W1=g(net[0].weight).T, b1=g(net[0].bias),
        W2=g(net[2].weight).T, b2=g(net[2].bias),
        W3=g(p.action_net.weight).T, b3=g(p.action_net.bias),
        obs_mean=venv.obs_rms.mean.copy(),
        obs_std=np.sqrt(venv.obs_rms.var + venv.epsilon),
    )


def evaluate(w, env, n=30, seed0=10_000):
    """Средний возврат детерминированной политики (через тот же act, что у жюри)."""
    total = 0.0
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        done, ret = False, 0.0
        while not done:
            obs, r, terminated, truncated, _ = env.step(act(obs, w))
            ret += r
            done = terminated or truncated
        total += ret
    return total / n


class EvalSave(BaseCallback):
    def __init__(self, venv, every=200_000, n_eval=30):
        super().__init__()
        self.venv, self.every, self.n_eval = venv, every, n_eval
        self.next_eval = every
        self.env = RobotWalker()
        self.best = -1.0

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if self.num_timesteps >= self.next_eval:
            self.next_eval = self.num_timesteps + self.every
            self.check()

    def check(self):
        # сохраняем и перечитываем файл, как это сделает жюри (float32 и т.д.)
        save_submission("_candidate.npz", **export_weights(self.model, self.venv))
        R = evaluate(load_submission("_candidate.npz"), self.env, self.n_eval)
        score = 100 * min(1.0, R / MAX_RETURN)
        if score > self.best:
            self.best = score
            shutil.copyfile("_candidate.npz", "submission.npz")
        self.model.save("ppo_model")
        self.venv.save("vecnorm.pkl")
        print(f"[eval] шагов={self.num_timesteps}  R={R:.1f}  score={score:.2f}  "
              f"лучший={self.best:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--envs", type=int, default=min(os.cpu_count() or 2, 16))
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    vec_cls = SubprocVecEnv if a.envs > 1 else DummyVecEnv
    raw = make_vec_env(RobotWalker, n_envs=a.envs, seed=0, vec_env_cls=vec_cls)

    resume = a.resume and os.path.exists("ppo_model.zip") and os.path.exists("vecnorm.pkl")
    if resume:
        venv = VecNormalize.load("vecnorm.pkl", raw)
        model = PPO.load("ppo_model", env=venv, device="cpu")
    else:
        venv = VecNormalize(raw, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=GAMMA)
        model = PPO(
            "MlpPolicy", venv,
            n_steps=BATCH_TOTAL // a.envs, batch_size=1024, n_epochs=10,
            gamma=GAMMA, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
            learning_rate=lambda f: 3e-4 * (0.1 + 0.9 * f),
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                               activation_fn=nn.Tanh, log_std_init=-1.0),
            device="cpu", verbose=1, seed=0,
        )

    cb = EvalSave(venv)
    model.learn(a.steps, callback=cb, reset_num_timesteps=not resume, progress_bar=False)
    cb.check()
    print(f"Готово. Лучший score по локальной оценке: {cb.best:.2f}. Файл: submission.npz")


if __name__ == "__main__":
    main()