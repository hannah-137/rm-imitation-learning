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
    kind = "success" if infos[-1]["success"] else "failure"
    make_demo_video(states, f"teacher_demo_videos/teacher_demo_seed{seed}_{kind}")
    if infos[-1]["success"] and len(success_demos) < 4:
        success_demos.append(states)
demos = pd.DataFrame(infos)
demos.to_csv("results/demo_collection_results/teacher_demo_success_distance_length_per_episode.csv", index=False)

successful = demos[demos["success"]]
stats = {
    "num_trajectories": len(demos),
    "length_mean": demos["length"].mean(),
    "length_std": demos["length"].std(),
    "success_fraction": demos["success"].mean(),
    "first_step_near_target_mean": successful["first_step_near_target"].mean(),
    "first_step_near_target_std": successful["first_step_near_target"].std(),
}
json.dump(stats, open("results/demo_collection_results/teacher_demo_count_length_success_rate.json", "w"), indent=2)
demo_config = {
    "demo_seeds": demo_seeds,
    "dataset_sizes": [100, 25],
}
json.dump(demo_config, open("configs/demo_collection_configs/teacher_demo_collection_seeds_and_dataset_sizes.json", "w"), indent=2)

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
        root=f"lerobot_datasets_for_act_training/lerobot_dataset_for_act_training_first_{num_demos}_teacher_demos",
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
strips = []
for states in success_demos:
    strips.append(make_demo_image(states))
media.write_image("results/demo_collection_results/teacher_demo_frames_first_4_successes.png", np.concatenate(strips, axis=0))
print("collect demos finished")