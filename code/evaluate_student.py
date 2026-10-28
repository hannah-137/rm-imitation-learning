from utils import eval_seeds, run_eval_policy
import sys
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
import numpy as np
import pandas as pd

def student(obs):
    o = torch.tensor(np.asarray(obs))
    batch = preprocess({"observation.state": o, "observation.environment_state": o})
    action = postprocess(policy.select_action(batch))
    return np.clip(action[0].numpy(), -1.0, 1.0)

teacher_success = pd.read_csv("outputs/eval_teacher.csv")["success"].mean()
rows = []
for num_demos in sys.argv[1:]:
    name = f"act_{num_demos}"
    path = f"models/{name}/checkpoints/last/pretrained_model"
    policy = ACTPolicy.from_pretrained(path, strict=True)
    preprocess, postprocess = make_pre_post_processors(policy.config, pretrained_path=path)
    student_success = run_eval_policy(student, name, reset_policy=policy.reset)
    rows.append({
        "condition": name,
        "num_demos": int(num_demos),
        "teacher_success": teacher_success,
        "student_success": student_success,
        "eval_episodes": len(eval_seeds),
    })
pd.DataFrame(rows).to_csv("outputs/summary.csv", index=False)
print(pd.DataFrame(rows))
