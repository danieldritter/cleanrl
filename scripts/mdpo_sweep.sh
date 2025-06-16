#!/bin/env bash

#SBATCH --job-name=mujoco_no_anneal_run
#SBATCH -D .
#SBATCH --output=./slurm/%x_%j.out
#SBATCH -e ./slurm/%x_%j.err
#SBATCH --mail-user=danieldritter1@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=sapphire
#SBATCH --mem=32GB
#SBATCH --time=0-4:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-5

environment_values=( HalfCheetah-v4 Walker2d-v4 Hopper-v4 Humanoid-v4 HumanoidStandup-v4 Ant-v4 )
# seed_values=( 15 47 274 )
seed_values=( 0 )

trial=${SLURM_ARRAY_TASK_ID}
environment=${environment_values[$(( trial % ${#environment_values[@]} ))]}
trial=$(( trial / ${#environment_values[@]} ))
seed=${seed_values[$(( trial % ${#seed_values[@]} ))]}
wandb_group="no_annealing_sweeps"
ETA=0.5
experiment_name="mdpo-eta_$ETA-$environment"
module load python
conda activate cleanrl 
echo which python 
## use ${environment}, ${seed} below

# default cleanrl config run  mdpo/klppo
python cleanrl/ppo_continuous_action.py \
    --exp_name=$experiment_name \
    --wandb_group=$wandb_group \
    --track \
    --env_id=$environment \
    --seed=$seed \
    --wandb_project_name="MujocoRuns" \
    --capture_video \
    --total_timesteps=1000000 \
    --use_kl_penalty \
    --kl_penalty_coef=$ETA \
    --kl_direction="reverse" \
    --no_anneal_lr \
    --no_normalize_reward 


# default cleanrl config run for ppo 
# python cleanrl/ppo_continuous_action.py \
#     --exp_name=$experiment_name \
#     --wandb_group=$wandb_group \
#     --track \
#     --env_id=$environment \
#     --seed=$seed \
#     --wandb_project_name="MujocoRuns" \
#     --capture_video \
#     --total_timesteps=1000000

# minimal config from MDPO 
# python cleanrl/ppo_continuous_action.py \
#     --exp_name=$experiment_name \
#     --wandb_group=$wandb_group \
#     --track \
#     --env_id=$environment \
#     --seed=$seed \
#     --wandb_project_name="MujocoRuns" \
#     --capture_video \
#     --total_timesteps=1000000 \
#     --num_minibatches=1 \
#     --vf_num_minibatches=16 \
#     --update_epochs=5 \
#     --vf_update_epochs=1 \
#     --ent_coef=0.0 \
#     --no_clip_vloss \
#     --gae_lambda=1.0 \
#     --no_anneal_lr \
#     --no_norm_adv \
#     --no_normalize_obs \
#     --no_normalize_reward \
#     --no_init_weight_orthogonal \
#     --use_kl_penalty \
#     --kl_penalty_coef=0.0 \
#     --kl_direction="reverse"

# minimal config from PPO
# python cleanrl/ppo_continuous_action.py \
#     --exp_name=$experiment_name \
#     --wandb_group=$wandb_group \
#     --track \
#     --env_id=$environment \
#     --seed=$seed \
#     --wandb_project_name="MujocoRuns" \
#     --capture_video \
#     --total_timesteps=10000000 \
#     --num_minibatches=32 \
#     --vf_num_minibatches=32 \
#     --update_epochs=1 \
#     --vf_update_epochs=1 \
#     --ent_coef=0.0 \
#     --no_clip_vloss \
#     --gae_lambda=1.0 \
#     --no_anneal_lr \
#     --no_norm_adv \
#     --no_normalize_obs \
#     --no_normalize_reward \
#     --no_init_weight_orthogonal
