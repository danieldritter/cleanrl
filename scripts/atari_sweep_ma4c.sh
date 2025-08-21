#!/bin/env bash

#SBATCH --job-name=ma4c_atari
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
#SBATCH --array=0-3

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
# CLIP_COEFS=(1.0 2.0)
# CLIP_COEFS=( 1.0 )
CLIP_COEFS=( 0.1 )
LEAK_COEFS=( 0.0 0.1 0.3 0.5 0.7 )
CLIP_COEF=${CLIP_COEFS[$((INDEX % ${#CLIP_COEFS[@]}))]}
INDEX=$(( INDEX / ${#CLIP_COEFS[@]} ))
LEAK_COEF=${LEAK_COEFS[$((INDEX % ${#LEAK_COEFS[@]}))]}
wandb_group="adv_clipping_runs"
# UPDATE_TYPE="half_std_clamp"
# UPDATE_TYPE="trefree"
UPDATE_TYPE="leaky_ppo"
# UPDATE_TYPE="ppo"
# UPDATE_TYPE="center_std_clamp"
ENT_COEF=0.01
experiment_name="$UPDATE_TYPE-ent$ENT_COEF-_std$CLIP_COEF-$environment"
module load python
conda activate cleanrl 
echo which python 
python cleanrl/ma4c_atari_clipped.py \
    --exp_name=$experiment_name \
    --wandb_group=$wandb_group \
    --track \
    --env_id=$environment \
    --wandb_project_name="AtariRuns" \
    --capture_video \
    --cuda \
    --clip_coef=$CLIP_COEF \
    --update_type=$UPDATE_TYPE \
    --ent_coef=$ENT_COEF \
    --leak_coef=$LEAK_COEF