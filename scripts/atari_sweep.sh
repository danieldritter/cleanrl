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
#SBATCH --array=0-1
# #SBATCH --array=0-3

# environments=(
#     "ALE/StarGunner-v5"
#     "ALE/DemonAttack-v5"
#     "ALE/BankHeist-v5"
#     "ALE/Alien-v5"
# )
environments=(
    "ALE/Gravitar-v5"
    "ALE/Freeway-v5"
    )
environment=${environments[$SLURM_ARRAY_TASK_ID]}
wandb_group="no_anneal_test"
experiment_name="mdpo-eta_1.0-$environment"


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
    --no_anneal_lr \
    --use_kl_penalty \
    --kl_penalty_coef=1 \
    --kl_direction="reverse"

