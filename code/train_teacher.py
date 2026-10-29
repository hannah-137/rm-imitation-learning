from utils import env, eval_seeds, env_name, impl, run_eval_policy, load_teacher, teacher_path
import functools
import time
import jax
import json
from brax.training.agents.ppo import networks as ppo_networks
# from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from absl import flags
from brax.io import model
from mujoco_playground import wrapper
from mujoco_playground.config import manipulation_params
from mujoco_playground import registry

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
model.save_params(teacher_path, theta)
print("ppo teacher train finished")

# evaluate ppo teacher and save results
teacher = load_teacher()
teacher_success = run_eval_policy(teacher, "teacher")
config = {
    "env": "PandaPickCube",
    "impl": "jax",
    "env_config": registry.get_default_config("PandaPickCube").to_dict(),
    "ppo_config": ppo_params.to_dict(),
    "ppo_seed": ppo_seed,
    "eval_seeds": eval_seeds,
    "teacher_train_minutes": ppo_minutes,
    "ppo_progress": progress,
    "teacher_eval_success": teacher_success
}
json.dump(config, open("records/teacher_summary.json", "w"), indent=2)
print(json.dumps(config, indent=2))
print("ppo teacher evaluate finished")