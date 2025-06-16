#!/bin/env bash

#SBATCH --job-name=mujoco_sweep
#SBATCH -D .
#SBATCH --output=./slurm/%x_%j.out
#SBATCH -e ./slurm/%x_%j.err
#SBATCH --mail-user=danieldritter1@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=sapphire
#SBATCH --mem=32GB
#SBATCH --time=0-4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-11
# #SBATCH --array=0-35
# #SBATCH --array=0-29
# #SBATCH --array=0-29
environment_values=( HumanoidStandup-v4 Humanoid-v4 Walker2d-v4 Hopper-v4 HalfCheetah-v4 Ant-v4 )
# seed_values=( 1 2 3 4 5 )
seed_values=( 1 2 )


trial=${SLURM_ARRAY_TASK_ID}
environment=${environment_values[$(( trial % ${#environment_values[@]} ))]}
trial=$(( trial / ${#environment_values[@]} ))
seed=${seed_values[$(( trial % ${#seed_values[@]} ))]}
wandb_group="discretized"
# wandb_group="mdpo_cleanrl_short"
experiment_name="mdpo-$environment"


module load python
conda activate cleanrl 
echo which python 
## use ${environment}, ${seed} below

python cleanrl/ppo_continuous_action.py \
    --exp_name=$experiment_name \
    --wandb_group=$wandb_group \
    --env_id=$environment \
    --seed=$seed \
    --wandb_project_name="MujocoRuns" \
    --capture_video \
    --kl_estimator="low_var" \
    --no_cuda \
    --use_kl_penalty \
    --mdpo_anneal_kl_penalty \
    --kl_direction="reverse" \
    --track \
    --is_discretized \
