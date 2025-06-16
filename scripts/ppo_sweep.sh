#!/bin/env bash

#SBATCH --job-name=mujoco_sweep
#SBATCH -D .
#SBATCH --output=./slurm/%x_%j.out
#SBATCH -e ./slurm/%x_%j.err
#SBATCH --mail-user=danieldritter1@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=kempner_h100
#SBATCH --mem=80GB
#SBATCH --time=0-12:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --array=0-16
# #SBATCH --array=17

environment_values=( HalfCheetah-v4 Walker2d-v4 Hopper-v4 InvertedPendulum-v4 Humanoid-v4 Pusher-v4 )
seed_values=( 15 47 274 )

trial=${SLURM_ARRAY_TASK_ID}
environment=${environment_values[$(( trial % ${#environment_values[@]} ))]}
trial=$(( trial / ${#environment_values[@]} ))
seed=${seed_values[$(( trial % ${#seed_values[@]} ))]}
wandb_group="mujoco_sweeps"
experiment_name="$environment-$seed"
module load python
conda activate cleanrl 
echo which python 
## use ${environment}, ${seed} below

python cleanrl/ppo_continuous_action.py \
    --exp_name=$experiment \
    --wandb_group=$wandb_group \
    --track \
    --env_id=$environment \
    --seed=$seed \
    --wandb_project_name="MujocoRuns" \
    --capture_video \
    --total_timesteps=1000000