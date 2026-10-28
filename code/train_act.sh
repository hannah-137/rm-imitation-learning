set -e
N=$1
lerobot-train \
  --dataset.repo_id=local/pick_cube_$N \
  --dataset.root=data/lerobot_$N \
  --policy.type=act \
  --policy.chunk_size=20 \
  --policy.n_action_steps=10 \
  --policy.normalization_mapping='{"STATE": "MEAN_STD", "ENV": "MEAN_STD", "ACTION": "MEAN_STD"}' \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --output_dir=models/act_$N \
  --job_name=act_$N \
  --save_freq=100000 \
  --wandb.enable=true \
  --wandb.project=imitation-learning \
  --wandb.disable_artifact=true
cp models/act_$N/checkpoints/last/pretrained_model/train_config.json records/act_$N.json
