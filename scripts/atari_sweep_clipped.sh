#!/bin/env bash

#SBATCH --job-name=atari_test
#SBATCH -D .
#SBATCH --output=./slurm/%x_%j.out
#SBATCH -e ./slurm/%x_%j.err
#SBATCH --mail-user=danieldritter1@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --mem=48GB
#SBATCH --time=0-18:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --partition=kempner_h100
#SBATCH --array=0,1,3
# #SBATCH --array=0-3
# #SBATCH --array=0-3

# environments=(
#     "ALE/StarGunner-v5"
#     "ALE/DemonAttack-v5"
#     "ALE/BankHeist-v5"
#     "ALE/Alien-v5"
# )
environments=(
    "ALE/Alien-v5"
    "ALE/BeamRider-v5"
    "ALE/Amidar-v5"
    "ALE/DemonAttack-v5"
    )
environment=${environments[$SLURM_ARRAY_TASK_ID]}
wandb_group="hard_clip_runs"
CLIP_COEF_LOWER=-0.1
CLIP_COEF_UPPER=0.1
ETA=1.0
experiment_name="reverse_rerun_clip_0.1_squared_kl_ent_0.01-$environment"
# experiment_name=ppo_clip_0.1_ent_0.01-$environment
ENT_COEF=0.01

module load python
conda activate cleanrl 
echo which python 
python cleanrl/ppo_atari_clipped.py \
    --exp_name=$experiment_name \
    --wandb_group=$wandb_group \
    --track \
    --env_id=$environment \
    --wandb_project_name="AtariRuns" \
    --capture_video \
    --cuda \
    --use_kl_penalty \
    --kl_estimator "squared" \
    --kl_penalty_coef=$ETA \
    --kl_ratio_clip_coef_lower=$CLIP_COEF_LOWER \
    --kl_ratio_clip_coef_upper=$CLIP_COEF_UPPER \
    --kl_direction="reverse" \
    --ent_coef=$ENT_COEF

# python cleanrl/ppo_atari_clipped.py \
#     --exp_name=$experiment_name \
#     --wandb_group=$wandb_group \
#     --track \
#     --env_id=$environment \
#     --wandb_project_name="AtariRuns" \
#     --capture_video \
#     --cuda \
#     --kl_estimator "standard" \
#     --ent_coef=$ENT_COEF \
#     --clip_coef=0.1

