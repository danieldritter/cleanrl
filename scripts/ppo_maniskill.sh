#!/bin/env bash

#SBATCH --job-name=maniskill_test
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
environment="AssemblingKits-v1"
# environment="DrawTriangle-v1"
wandb_group="cleanrl_default"
experiment_name="ppo-$environment"


module load python
conda activate cleanrl 
echo which python 

NUM_STEPS=200
NUM_TRAIN_ENVS=512
NUM_EVAL_ENVS=8
TOTAL_TIMESTEPS=50000000
# TOTAL_TIMESTEPS=20000000
LR=0.001
UPDATE_EPOCHS=5
python cleanrl/ppo_mani_skill.py \
    --exp_name=$experiment_name \
    --track \
    --env_id=$environment \
    --wandb_project_name="ManiSkillRuns" \
    --capture_video \
    --num_steps=$NUM_STEPS \
    --num_eval_steps=$NUM_STEPS \
    --num_envs=$NUM_TRAIN_ENVS \
    --num_eval_envs=$NUM_EVAL_ENVS \
    --total_timesteps=$TOTAL_TIMESTEPS \
    --learning_rate=$LR \
    --update_epochs=$UPDATE_EPOCHS \
    --clip_vloss \
    --gamma=0.95 \
    --gae_lambda=0.95 \
    --cuda 
