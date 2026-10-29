from utils import run, get_episode_info, make_demo_image, make_demo_video, load_teacher, demo_seeds
import jax
import numpy as np
import pandas as pd
import mediapy as media
import json
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# collect demos
teacher = load_teacher()
episodes, infos, success_demos = [], [], []
for seed in demo_seeds:
    states, obs, actions, rewards = run(seed, teacher)
    episodes.append([obs, actions, rewards])
    infos.append(get_episode_info(seed, obs, actions, states[-1]))
    if infos[-1]["success"] and len(success_demos) < 4:
        success_demos.append(states)
demos = pd.DataFrame(infos)
demos.to_csv("outputs/demos.csv", index=False)

successful = demos[demos["success"]]
stats = {
    "demo_seeds": demo_seeds,
    "num_trajectories": len(demos),
    "length_mean": demos["length"].mean(),
    "length_std": demos["length"].std(),
    "success_fraction": demos["success"].mean(),
    "first_step_near_target_mean": successful["first_step_near_target"].mean(),
    "first_step_near_target_std": successful["first_step_near_target"].std(),
}
json.dump(stats, open("outputs/demo_summary.json", "w"), indent=2)

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
print("collect demos finished")