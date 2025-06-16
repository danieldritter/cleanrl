#!/bin/bash
#SBATCH -J mujoco_test                         # Job name
#SBATCH -D .
#SBATCH -o ./slurm/%x_%j.out                  # output file (%j expands to jobID)
#SBATCH -e ./slurm/%x_%j.err                  # error log file (%j expands to jobID)
#SBATCH --mail-type=ALL                      # Request status by email 
#SBATCH --mail-user=danieldritter1@gmail.com        # Email address to send results to.
#SBATCH --mem=80G                          # server memory requested (per node)
#SBATCH -c 8
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH -t 24:00:00 # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:1
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --partition=kempner_h100
module load python
conda activate cleanrl 
which python

python cleanrl/ppo_continuous_action.py \
    --exp_name="ppo_mujoco_test" \
    --track \
    --env_id="HalfCheetah-v4" \
    --wandb_project_name="MujocoRuns" \
    --capture_video \
    --total_timesteps=1000000

# python -m cleanrl_utils.benchmark \
#     --env-ids HalfCheetah-v4 Walker2d-v4 Hopper-v4 InvertedPendulum-v4 Humanoid-v4 Pusher-v4 \
#     --command "poetry run python cleanrl/ppo_continuous_action.py --no_cuda --track --capture_video" \
#     --num-seeds 3 \
#     --workers 9 \
#     --slurm-gpus-per-task 1 \
#     --slurm-ntasks 1 \
#     --slurm-total-cpus 10 \
#     --slurm-template-path benchmark/cleanrl_1gpu.slurm_template

# python -m cleanrl_utils.benchmark \
#     --env-ids dm_control/acrobot-swingup-v0 dm_control/acrobot-swingup_sparse-v0 dm_control/ball_in_cup-catch-v0 dm_control/cartpole-balance-v0 dm_control/cartpole-balance_sparse-v0 dm_control/cartpole-swingup-v0 dm_control/cartpole-swingup_sparse-v0 dm_control/cartpole-two_poles-v0 dm_control/cartpole-three_poles-v0 dm_control/cheetah-run-v0 dm_control/dog-stand-v0 dm_control/dog-walk-v0 dm_control/dog-trot-v0 dm_control/dog-run-v0 dm_control/dog-fetch-v0 dm_control/finger-spin-v0 dm_control/finger-turn_easy-v0 dm_control/finger-turn_hard-v0 dm_control/fish-upright-v0 dm_control/fish-swim-v0 dm_control/hopper-stand-v0 dm_control/hopper-hop-v0 dm_control/humanoid-stand-v0 dm_control/humanoid-walk-v0 dm_control/humanoid-run-v0 dm_control/humanoid-run_pure_state-v0 dm_control/humanoid_CMU-stand-v0 dm_control/humanoid_CMU-run-v0 dm_control/lqr-lqr_2_1-v0 dm_control/lqr-lqr_6_2-v0 dm_control/manipulator-bring_ball-v0 dm_control/manipulator-bring_peg-v0 dm_control/manipulator-insert_ball-v0 dm_control/manipulator-insert_peg-v0 dm_control/pendulum-swingup-v0 dm_control/point_mass-easy-v0 dm_control/point_mass-hard-v0 dm_control/quadruped-walk-v0 dm_control/quadruped-run-v0 dm_control/quadruped-escape-v0 dm_control/quadruped-fetch-v0 dm_control/reacher-easy-v0 dm_control/reacher-hard-v0 dm_control/stacker-stack_2-v0 dm_control/stacker-stack_4-v0 dm_control/swimmer-swimmer6-v0 dm_control/swimmer-swimmer15-v0 dm_control/walker-stand-v0 dm_control/walker-walk-v0 dm_control/walker-run-v0 \
#     --command "poetry run python cleanrl/ppo_continuous_action.py --exp-name ppo_continuous_action_8M  --total-timesteps 8000000 --no_cuda --track" \
#     --num-seeds 10 \
#     --workers 9 \
#     --slurm-gpus-per-task 1 \
#     --slurm-ntasks 1 \
#     --slurm-total-cpus 10 \
#     --slurm-template-path benchmark/cleanrl_1gpu.slurm_template


