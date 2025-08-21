#!/bin/env bash

#SBATCH --job-name=mujoco_sppo
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
#SBATCH --array=2-59
# #SBATCH --array=0-59

# environment_values=( HalfCheetah-v4 Walker2d-v4 Hopper-v4 Humanoid-v4 HumanoidStandup-v4 Ant-v4 )
environment_values=( HalfCheetah-v4 Walker2d-v4 Hopper-v4 Humanoid-v4 Ant-v4 )
seed_values=( 15 47 274 )
coef_values=( 0.1 0.3 0.5 0.7)
# seed_values=( 0 )

trial=${SLURM_ARRAY_TASK_ID}
environment=${environment_values[$(( trial % ${#environment_values[@]} ))]}
trial=$(( trial / ${#environment_values[@]} ))
seed=${seed_values[$(( trial % ${#seed_values[@]} ))]}
trial=$(( trial / ${#seed_values[@]} ))
coef=${coef_values[$(( trial % ${#coef_values[@]} ))]}
num_updates=10
wandb_group="sppo_initial_runs"
experiment_name="ppo-$environment-seed-$seed-coef-$coef-updates-$num_updates"
module load python
conda activate cleanrl 
echo which python 
## use ${environment}, ${seed} below

# default cleanrl config run  mdpo/klppo
python cleanrl/ppo_continuous_action_clipped.py \
    --exp_name=$experiment_name \
    --no_cuda \
    --wandb_group=$wandb_group \
    --track \
    --env_id=$environment \
    --seed=$seed \
    --wandb_project_name="MujocoSPPO" \
    --capture_video \
    --clip_coef=$coef \
    --update_epochs=$num_updates \
    --no_norm_adv
    # --use_spma


