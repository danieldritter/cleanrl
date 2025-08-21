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
#SBATCH --array=0-19
# #SBATCH --array=0
# #SBATCH --array=0-3

environments=(
    "ALE/Alien-v5"
    "ALE/BeamRider-v5"
    "ALE/Amidar-v5"
    "ALE/DemonAttack-v5"
    )
ETAS=(0.2 0.6 0.8 1.4 1.8)

trial=${SLURM_ARRAY_TASK_ID}
environment=${environments[$(( trial % ${#environments[@]} ))]}
trial=$(( trial / ${#environments[@]} ))
ETA=${ETAS[$(( trial % ${#ETAS[@]} ))]}
wandb_group="eta_runs"
experiment_name="reverse-eta$ETA-$environment"
ENT_COEF=0.01

module load python
conda activate cleanrl 
echo which python 
python cleanrl/ppo_atari.py \
    --exp_name=$experiment_name \
    --wandb_group=$wandb_group \
    --track \
    --env_id=$environment \
    --wandb_project_name="AtariRuns" \
    --capture_video \
    --cuda \
    --use_kl_penalty \
    --kl_penalty_coef=$ETA \
    --kl_direction="reverse" \
    --ent_coef=$ENT_COEF

