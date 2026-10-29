
import os
os.environ["MUJOCO_GL"] = "egl"  
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" 
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"  

import jax
import functools
from mujoco_playground import registry
from mujoco_playground.config import manipulation_params
import numpy as np
import pandas as pd
import mediapy as media
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model

for folder in ["data","outputs","records","videos","models"]:
    os.makedirs(folder, exist_ok=True)

env_name = "PandaPickCube"
impl = "jax"
episode_length = 150
success_dist = 0.05
eval_seeds = list(range(1000, 1050))
demo_seeds = list(range(100))

env = registry.load(env_name, config_overrides={"impl": impl})
reset = jax.jit(env.reset)
step = jax.jit(env.step)

teacher_path = "data/teacher_params.pkl"
def load_teacher():
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"{teacher_path} not found. Run train_teacher.py first.")
    ppo_params = manipulation_params.brax_ppo_config(env_name, impl)
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **ppo_params.network_factory)
    network = network_factory(
        env.observation_size, 
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize
    )
    params = model.load_params(teacher_path)
    teacher_policy = jax.jit(ppo_networks.make_inference_fn(network)(params, deterministic=True))
    key = jax.random.PRNGKey(0)

    def teacher(obs):
        action, _ = teacher_policy(obs, key)
        return action
    return teacher

def run(seed, policy):
    state = reset(jax.random.PRNGKey(seed))
    states, obs, actions, rewards = [state], [], [], []
    for t in range(episode_length):
        action = np.asarray(policy(state.obs), dtype=np.float32)
        obs.append(np.asarray(state.obs))  
        actions.append(action)
        state = step(state, action)
        states.append(state)
        rewards.append(float(state.reward))
        if state.done:  
            break
    return states, np.array(obs), np.array(actions), np.array(rewards)

def get_episode_info(seed, obs, actions, last_state):
    final_dist = float(np.linalg.norm(np.asarray(last_state.obs)[49:52]))
    dists = np.append(np.linalg.norm(obs[:, 49:52], axis=1), final_dist)
    near_target = dists < success_dist
    target = np.asarray(last_state.info["target_pos"])
    return {
        "seed": seed,
        "box_x": float(obs[0, 9]),
        "box_y": float(obs[0, 10]),
        "target_x": float(target[0]),
        "target_y": float(target[1]),
        "target_z": float(target[2]),
        "length": len(actions),
        "success": final_dist < success_dist,
        "final_dist": final_dist,
        "reached_box": float(last_state.info["reached_box"]),
        "max_box_height": float(obs[:, 11].max()),
        "first_step_near_target": int(np.argmax(near_target)) if near_target.any() else -1,
        "mean_action_change": float(np.abs(np.diff(actions, axis=0)).mean()),
    }

def make_demo_video(states, name):
    frames = env.render(states[::2], height=480, width=640)
    media.write_video(f"videos/{name}.mp4", frames, fps=round(1.0 / env.dt / 2))
    media.write_image(f"outputs/{name}_frames.png", make_demo_image(states))

def make_demo_image(states):
    frames = env.render(states[::30], height=240, width=320)
    return np.concatenate(frames, axis=1)

def run_eval_policy(policy, name, reset_policy=None):
    rows = []
    saved = []
    for seed in eval_seeds:
        if reset_policy is not None:
            reset_policy()
        states, obs, actions, rewards = run(seed, policy)
        rows.append(get_episode_info(seed, obs, actions, states[-1]))
        kind = "success" if rows[-1]["success"] else "failure"
        if kind not in saved:
            make_demo_video(states, f"{name}_{kind}")
            saved.append(kind)
    table = pd.DataFrame(rows)
    table.to_csv(f"outputs/eval_{name}.csv", index=False)
    print(name, "success rate:", table["success"].mean())
    return table["success"].mean()

