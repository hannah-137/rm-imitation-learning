from utils import demo_seeds, eval_seeds, success_dist, env, run, get_episode_info, make_demo_image, make_demo_video, run_eval_policy
# import datetime
import functools

# import json
# import os
import time
# import warnings
import jax

from brax.training.agents.ppo import networks as ppo_networks
# from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from absl import flags
from brax.io import model
from mujoco_playground import registry
from mujoco_playground import wrapper
# from mujoco_playground.config import dm_control_suite_params
# from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import mediapy as media
import json

env_name = "PandaPickCube"
impl = "jax"
ppo_seed = 1

# ppo learning
ppo_params = manipulation_params.brax_ppo_config(env_name, impl)
training_params = dict(ppo_params)
if "network_factory" in training_params:
    del training_params["network_factory"]

progress = []
def log_progress(num_steps, metrics):
    reward = float(metrics["eval/episode_reward"])
    progress.append([num_steps, reward])
    print("ppo steps:", num_steps, "eval episode reward:", reward)

start = time.time()
pie, theta, _ = ppo.train(
    environment=env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    network_factory=functools.partial(ppo_networks.make_ppo_networks, **ppo_params.network_factory),
    progress_fn=log_progress,
    seed=ppo_seed,
    **training_params,
)
ppo_minutes = (time.time() - start) / 60
model.save_params("data/teacher_params.pkl", theta)
teacher_policy = jax.jit(pie(theta, deterministic=True))


# collect demos
key = jax.random.PRNGKey(0) 
def teacher(obs):
    action, _ = teacher_policy(obs, key)
    return action

episodes, infos, success_demos = [], [], []
for seed in demo_seeds:
    states, obs, actions, rewards = run(seed, teacher)
    episodes.append([obs, actions, rewards])
    infos.append(get_episode_info(seed, obs, actions, states[-1]))
    if infos[-1]["success"] and len(success_demos) < 4:
        success_demos.append(states)
demos = pd.DataFrame(infos)
demos.to_csv("outputs/demos.csv", index=False)


# demos to LeRobot datasets
features = {
    "observation.state": {"dtype": "float32", "shape": (66,), "names": None},
    "observation.environment_state": {"dtype": "float32", "shape": (66,), "names": None},
    "action": {"dtype": "float32", "shape": (8,), "names": None},
    "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
}
for num_demos in [100, 25]:
    dataset = LeRobotDataset.create(
        f"local/pick_cube_{num_demos}",
        fps=50,
        features=features,
        root=f"data/lerobot_{num_demos}",
        use_videos=False
    )
    for obs, actions, rewards in episodes[:num_demos]:
        for o, a, r in zip(obs, actions, rewards):
            frame = {"observation.state": o, "observation.environment_state": o, "action": a}
            frame["next.reward"] = np.array([r], dtype=np.float32)
            frame["task"] = "pick up the box and move it to the target"
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()

# save demo videos and images
make_demo_video(success_demos[0], "teacher_demo")
strips = []
for states in success_demos:
    strips.append(make_demo_image(states))
media.write_image("outputs/demo_frames.png", np.concatenate(strips, axis=0))

# evaluate teacher and save results
teacher_success = run_eval_policy(teacher, "teacher")
successful = demos[demos["success"]]
stats = {
    "num_trajectories": len(demos),
    "length_mean": demos["length"].mean(),
    "length_std": demos["length"].std(),
    "success_fraction": demos["success"].mean(),
    "first_step_near_target_mean": successful["first_step_near_target"].mean(),
    "first_step_near_target_std": successful["first_step_near_target"].std(),
    "teacher_eval_success": teacher_success,
    "teacher_train_minutes": ppo_minutes,
    "ppo_progress": progress,
}
json.dump(stats, open("outputs/dataset_stats.json", "w"), indent=2)
config = {
    "env": "PandaPickCube",
    "impl": "jax",
    "env_config": registry.get_default_config("PandaPickCube").to_dict(),
    "ppo_config": ppo_params.to_dict(),
    "ppo_seed": ppo_seed,
    "demo_seeds": demo_seeds,
    "eval_seeds": eval_seeds,
    "success_dist": success_dist,
}
json.dump(config, open("records/teacher.json", "w"), indent=2)
print(json.dumps(stats, indent=2))