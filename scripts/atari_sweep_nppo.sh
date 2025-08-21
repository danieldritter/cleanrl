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
# #SBATCH --array=0-19

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
INDEX=$SLURM_ARRAY_TASK_ID
environment=${environments[$((INDEX % ${#environments[@]}))]}
INDEX=$(( INDEX / ${#environments[@]} ))
CLIP_COEFS=(0.05 0.1 0.2 0.5 1.0)
CLIP_COEF=${CLIP_COEFS[$((INDEX % ${#CLIP_COEFS[@]}))]}
wandb_group="log_clip_sweep"
experiment_name="vf_clip_spma_reverse_$CLIP_COEF-$environment"
ENT_COEF=0.01

module load python
conda activate cleanrl 
echo which python 
python cleanrl/nppo_atari_clipped.py \
    --exp_name=$experiment_name \
    --wandb_group=$wandb_group \
    --track \
    --env_id=$environment \
    --wandb_project_name="AtariRuns" \
    --capture_video \
    --cuda \
    --clip_coef=$CLIP_COEF \
    --kl_direction="reverse" \
    --ent_coef=$ENT_COEF \
    --use_spma