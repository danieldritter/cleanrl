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
#SBATCH --array=0-10
# #SBATCH --array=4,269
# #SBATCH --array=0-269
# #SBATCH --array=0,1259
# #SBATCH --array=0-179
# #SBATCH --array=0-1259
environment_values=( HumanoidStandup-v5 Humanoid-v5 Walker2d-v5 Hopper-v5 HalfCheetah-v5 Ant-v5 )
# update_types=( "ppo" "advantage_pessimism" "advantage_optimism" "advantage_neutral" "advantage_clipped" "advantage_pessimism_clipped" "advantage_optimism_clipped" )
# update_types=( "ppo" )
# update_types=( "ppo" "advantage_neutral_spma" "advantage_clipped_spma" "advantage_clipped_spma_clipped" "advantage_neutral_spma_clipped" "advantage_pessimism" "advantage_pessimism" "advantage_optimism" "advantage_neutral" "advantage_clipped" "advantage_pessimism_clipped" "advantage_optimism_clipped" )
update_types=( "advantage_partial_pessimism_spma" "advantage_partial_pessimism_spma_clipped" "ppo" "advantage_neutral_spma" "advantage_clipped_spma" "advantage_clipped_spma_clipped" "advantage_neutral_spma_clipped" "advantage_pessimism" "advantage_pessimism" "advantage_optimism" "advantage_neutral" "advantage_clipped" "advantage_pessimism_clipped" "advantage_optimism_clipped" )
# update_types=( "advantage_partial_pessimism_spma_clipped" "advantage_clipped_spma_clipped" "advantage_neutral_spma_clipped" )
# seed_values=( 1 2 3 4 5 )
seed_values=( 17 )
clip_coefs=( 0.1 0.2 0.3 )
# adv_clip_coefs=( 0.1 0.2 0.3 )
# adv_clip_coefs=( 0.7 1.0 1.3 )
adv_clip_coefs=( 1.0 )
entropy_coefs=( 0.0 0.0001 0.001 0.01 0.1 )

trial=${SLURM_ARRAY_TASK_ID}
trial=4
environment=${environment_values[$(( trial % ${#environment_values[@]} ))]}
trial=$(( trial / ${#environment_values[@]} ))
seed=${seed_values[$(( trial % ${#seed_values[@]} ))]}
trial=$(( trial / ${#seed_values[@]} ))
update_type=${update_types[$(( trial % ${#update_types[@]} ))]}
trial=$(( trial / ${#update_types[@]} ))
CLIP_COEF=${clip_coefs[$(( trial % ${#clip_coefs[@]} ))]}
trial=$(( trial / ${#clip_coefs[@]} ))
ADV_CLIP_COEF=${adv_clip_coefs[$(( trial % ${#adv_clip_coefs[@]} ))]}
ENT_COEF=${entropy_coefs[$(( trial % ${#entropy_coefs[@]} ))]}

# LEAK_COEFS=( 0.0 0.1 0.3 0.5 0.7)
# LEAK_COEF=${LEAK_COEFS[$(( trial % ${#LEAK_COEFS[@]} ))]}

wandb_group="v5_clip_sweep"
# wandb_group="mdpo_cleanrl_short"
experiment_name="$update_type-ent$ENT_COEF-clip$CLIP_COEF-adv_clip$ADV_CLIP_COEF-$environment"
alg_name="$update_type-ent$ENT_COEF-clip$CLIP_COEF-adv_clip$ADV_CLIP_COEF"

module load python
conda activate cleanrl 
echo which python 
## use ${environment}, ${seed} below

# python cleanrl/ma4c_continuous_action.py \
#     --exp_name=$experiment_name \
#     --wandb_group=$wandb_group \
#     --env_id=$environment \
#     --seed=$seed \
#     --wandb_project_name="MujocoRunsv2" \
#     --no_cuda \
#     --track \
#     --update_type=$update_type \
#     --clip_coef=$CLIP_COEF \
#     --adv_clip_coef=$ADV_CLIP_COEF \
#     --alg_name=$alg_name \
#     --ent_coef=$ENT_COEF
    # --autotune_clip_coef \
    # --target_clipfrac=0.2

python cleanrl/ppo_v5_continuous_action.py \
    --exp_name=$experiment_name \
    --env_id=$environment \
    --seed=$seed \
    --wandb_project_name="MujocoRunsv2" \
    --no_cuda \
    --track