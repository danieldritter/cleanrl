#!/bin/env bash

#SBATCH --job-name=mujoco_tuning
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
#SBATCH --array=0-35
# #SBATCH --array=1-70
# #SBATCH --array=0-71

# environment_values=( Walker2d-v4 Hopper-v4 Humanoid-v4 HumanoidStandup-v4 )
environment_values=( HalfCheetah-v4 Walker2d-v4 Hopper-v4 Humanoid-v4 HumanoidStandup-v4 Ant-v4 )
seed_values=( 73 95 )
LR_values=( 0.0001 0.0003 0.001 )
num_updates_values=( 5 10 20 )

# including if we want to sweep later 
gae_lambda=0.9 
discount_factor=0.99
trial=${SLURM_ARRAY_TASK_ID}
environment=${environment_values[$(( trial % ${#environment_values[@]} ))]}
trial=$(( trial / ${#environment_values[@]} ))
seed=${seed_values[$(( trial % ${#seed_values[@]} ))]}
trial=$(( trial / ${#seed_values[@]} ))
LR=${LR_values[$(( trial % ${#LR_values[@]} ))]}
trial=$(( trial / ${#LR_values[@]} ))
num_updates=${num_updates_values[$(( trial % ${#num_updates_values[@]} ))]}
loss_name="klppo"

module load python
conda activate cleanrl 
experiment_name="${loss_name}_${environment}_seed${seed}_lr${LR}_updates${num_updates}"
wandb_group="mdpo_tuning_sweep_05_12_2025"
## use ${environment}, ${seed}, ${LR}, ${num_updates}, ${discount_factor} below
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
    --mdpo_anneal_kl_penalty \
    --kl_direction="forward" \
    --gae_lambda=$gae_lambda \
    --gamma=$discount_factor \
    --update_epochs=$num_updates \
    --learning_rate=$LR \
    --no_cuda
