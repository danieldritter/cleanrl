#!/bin/env bash

#SBATCH --job-name=mujoco_sweep
#SBATCH -D .
#SBATCH --output=./slurm/%x_%j.out
#SBATCH -e ./slurm/%x_%j.err
#SBATCH --mail-user=danieldritter1@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=sapphire
#SBATCH --mem=32GB
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-29
# #SBATCH --array=0-29
environment_values=( HumanoidStandup-v4 Humanoid-v4 Walker2d-v4 Hopper-v4 HalfCheetah-v4 Ant-v4)
seed_values=( 1 2 3 4 5 )
env_lrs=( 0.0001 0.0001 0.0001 0.0003 0.0001 0.0001)
env_num_updates=( 10 10 10 10 20 20 )

trial=${SLURM_ARRAY_TASK_ID}
environment=${environment_values[$(( trial % ${#environment_values[@]} ))]}
lr=${env_lrs[$(( trial % ${#env_lrs[@]} ))]}
num_updates=${env_num_updates[$(( trial % ${#env_num_updates[@]} ))]}
trial=$(( trial / ${#environment_values[@]} ))
seed=${seed_values[$(( trial % ${#seed_values[@]} ))]}
# wandb_group="default_no_lr_anneal"
# wandb_group="gamma0.95_lambda0.9"
wandb_group="top_runs"
# wandb_group="mdpo_cleanrl_short"
experiment_name="klppo-$environment"

gae_lambda=0.9
# LR=0.0003
discount_factor=0.99
# update_epochs=5

module load python
conda activate cleanrl 
echo which python 
## use ${environment}, ${seed} below

python cleanrl/ppo_continuous_action.py \
    --exp_name=$experiment_name \
    --wandb_group=$wandb_group \
    --track \
    --env_id=$environment \
    --seed=$seed \
    --wandb_project_name="MujocoRuns" \
    --capture_video \
    --total_timesteps=1000000 \
    --gae_lambda=$gae_lambda \
    --gamma=$discount_factor \
    --learning_rate=$lr \
    --update_epochs=$num_updates \
    --use_kl_penalty \
    --mdpo_anneal_kl_penalty \
    --kl_direction="forward" \
    --no_cuda


# python cleanrl/ppo_continuous_action.py \
#     --exp_name=$experiment_name \
#     --wandb_group=$wandb_group \
#     --track \
#     --env_id=$environment \
#     --seed=$seed \
#     --wandb_project_name="MujocoRuns" \
#     --capture_video \
#     --total_timesteps=1000000 \
#     --use_kl_penalty \
#     --mdpo_anneal_kl_penalty \
#     --kl_penalty_coef=5 \
#     --kl_direction="reverse" \
#     --no_anneal_lr \
#     --no_cuda \
#     --no_normalize_obs \
#     --no_normalize_reward \
#     --no_init_weight_orthogonal \
#     --max_grad_norm=1e9 \
#     --no_clip_vloss
