#!/bin/env bash

#SBATCH --job-name=atari_test
#SBATCH -D .
#SBATCH --output=./slurm/%x_%j.out
#SBATCH -e ./slurm/%x_%j.err
#SBATCH --mail-user=danieldritter1@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --mem=48GB
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --partition=kempner_h100
# environment="ALE/Gravitar-v5"
# environment="ALE/Freeway-v5"
environment="ALE/Breakout-v5"
# environment="ALE/Tennis-v5"
# environment="ALE/MsPacman-v5"
wandb_group="atari_tests"
experiment_name="klppo-episode-end-$environment"


module load python
conda activate cleanrl 
echo which python 
sparsity_steps=-1
python cleanrl/ppo_atari.py \
    --exp_name=$experiment_name \
    --wandb_group=$wandb_group \
    --track \
    --env_id=$environment \
    --wandb_project_name="AtariRuns" \
    --capture_video \
    --sparsity_steps=$sparsity_steps \
    --cuda \
    --use_kl_penalty \
    --mdpo_anneal_kl_penalty \
    --kl_direction="forward"
