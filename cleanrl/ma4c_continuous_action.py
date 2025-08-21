# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

MAX_SCORES = {
    "HalfCheetah-v4": 6000,
    "Hopper-v4": 5000,
    "Walker2d-v4": 6000,
    "Humanoid-v4": 3000,
    "Ant-v4": 5000,
    "HumanoidStandup-v4": 200000,
}


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_group: str = None
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    vf_num_minibatches: int = 32
    update_epochs: int = 10
    """the K epochs to update the policy"""
    vf_update_epochs: int = 10
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    autotune_clip_coef: bool = False
    target_clipfrac: float = 0.2
    """the surrogate clipping coefficient"""
    adv_clip_coef: float = 0.2
    leak_coef: float = 0.0
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    normalize_obs: bool = True
    """whether to normalize the observation"""
    normalize_reward: bool = True
    """whether to normalize the reward"""
    init_weight_orthogonal: bool = True
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    update_type: str = "ppo"  # one of "mdpo", "ppo", "klppo", "zeroeta"
    alg_name: str = "default"


def make_env(
    env_id,
    idx,
    capture_video,
    run_name,
    gamma,
    normalize_reward=True,
    normalize_obs=True,
):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(
            env,
        )  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        if normalize_obs:
            env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -10, 10),
            env.observation_space,
        )
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def closed_form_gaussian_kl(
    forward_mean,
    forward_logstd,
    reverse_mean,
    reverse_logstd,
):
    kl = reverse_logstd - forward_logstd
    kl += (reverse_mean - forward_mean) ** 2 / torch.exp(reverse_logstd)
    kl += torch.exp(forward_logstd) / torch.exp(reverse_logstd)
    kl = kl.sum(axis=-1) - forward_mean.shape[-1]
    kl *= 0.5
    return kl


def layer_init(layer, std=np.sqrt(2), bias_const=0.0, orthogonal=True):
    if orthogonal:
        torch.nn.init.orthogonal_(layer.weight, std)
    else:
        torch.nn.init.xavier_uniform_(layer.weight, gain=std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, orthogonal=True):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
                orthogonal=orthogonal,
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64), orthogonal=orthogonal),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0, orthogonal=orthogonal),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
                orthogonal=orthogonal,
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64), orthogonal=orthogonal),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, np.prod(envs.single_action_space.shape)),
                std=0.01,
                orthogonal=orthogonal,
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape)),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            action_mean,
            action_logstd,
        )

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
            action_mean,
            action_logstd,
        )


def compute_low_var_kl(log_probs_p, log_probs_q):
    # compute KL(p || q)
    low_var_kl = log_probs_q - log_probs_p
    ratio = torch.exp(low_var_kl)
    kld = (ratio - low_var_kl - 1).contiguous()
    return kld


def compute_jsd(old_logprobs, new_logprobs):
    # Compute the Jensen-Shannon Divergence
    p = torch.exp(old_logprobs)
    q = torch.exp(new_logprobs)
    m = 0.5 * (p + q)
    kl_p_m = compute_low_var_kl(old_logprobs, torch.log(m))
    kl_q_m = compute_low_var_kl(new_logprobs, torch.log(m))
    jsd = 0.5 * (kl_p_m + kl_q_m)
    return jsd


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.vf_minibatch_size = int(args.batch_size // args.vf_num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_name = args.exp_name
    clip_coef = args.clip_coef
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.gamma,
                normalize_reward=args.normalize_reward,
                normalize_obs=args.normalize_obs,
            )
            for i in range(args.num_envs)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )
    agent = Agent(envs, orthogonal=args.init_weight_orthogonal).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs, *envs.single_observation_space.shape),
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs, *envs.single_action_space.shape),
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    action_means = torch.zeros(
        (args.num_steps, args.num_envs, *envs.single_action_space.shape),
    ).to(device)
    action_logstds = torch.zeros(
        (args.num_steps, args.num_envs, np.prod(envs.single_action_space.shape)),
    ).to(device)
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    running_reward = None
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, action_mean, action_logstd = (
                    agent.get_action_and_value(next_obs)
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            action_means[step] = action_mean
            action_logstds[step] = action_logstd
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy(),
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            if "final_info" in infos:
                episodes_over = np.nonzero(infos["final_info"]["_episode"])[0]
                episodic_returns = infos["final_info"]["episode"]["r"][episodes_over]
                episodic_lengths = infos["final_info"]["episode"]["l"][episodes_over]
                for episodic_return, episodic_length in zip(
                    episodic_returns, episodic_lengths
                ):
                    print(
                        f"global_step={global_step}, episodic_return={episodic_return}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", episodic_return, global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", episodic_length, global_step
                    )
                    normalized_ep_return = episodic_return / MAX_SCORES.get(
                        args.env_id,
                        1.0,
                    )
                    running_reward = (
                        episodic_return
                        if running_reward is None
                        else 0.05 * episodic_return + 0.95 * running_reward
                    )
                    normalized_running_reward = running_reward / MAX_SCORES.get(
                        args.env_id,
                        1.0,
                    )
                    print(
                        f"iter={iteration}/{args.num_iterations}, global_step={global_step}, "
                        f"episodic_return={episodic_return}, running_reward={running_reward}",
                    )
                    writer.add_scalar(
                        "charts/running_reward",
                        running_reward,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/normalized_running_reward",
                        normalized_running_reward,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_return_normalized",
                        normalized_ep_return,
                        global_step,
                    )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1, *envs.single_observation_space.shape))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, *envs.single_action_space.shape))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_action_means = action_means.reshape(
            (-1, *envs.single_action_space.shape),
        ).detach()
        b_action_logstds = action_logstds.reshape(
            (-1, np.prod(envs.single_action_space.shape)),
        ).detach()
        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        grad_norms = []
        ratios = []
        max_ratios = []
        ratios_stds = []
        logratios = []
        logratios_stds = []
        max_logratios = []
        advantages = []
        max_advantages = []
        advantages_stds = []
        advantages_norm = []
        max_advantages_norm = []
        advantage_clipfrac = []
        for epoch in range(args.update_epochs):
            all_ratios = []
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, new_action_mean, new_action_logstd = (
                    agent.get_action_and_value(
                        b_obs[mb_inds],
                        b_actions[mb_inds],
                    )
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                all_ratios.append(ratio)
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    if "spma" not in args.update_type:
                        clipfracs += [
                            ((ratio - 1.0).abs() > clip_coef).float().mean().item(),
                        ]
                    else:
                        clipfracs += [
                            (torch.abs(logratio) > clip_coef).float().mean().item(),
                        ]

                mb_advantages = b_advantages[mb_inds]
                advantages += [mb_advantages.mean().item()]
                advantages_stds += [mb_advantages.std().item()]
                max_advantages += [mb_advantages.max().item()]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                if args.update_type == "ppo":
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio,
                        1 - clip_coef,
                        1 + clip_coef,
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                elif args.update_type == "neutral_ppo":
                    pg_loss = (
                        -mb_advantages
                        * torch.clamp(
                            ratio,
                            1 - clip_coef,
                            1 + clip_coef,
                        )
                    ).mean()
                elif args.update_type == "optimistic_ppo":
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio,
                        1 - clip_coef,
                        1 + clip_coef,
                    )
                    pg_loss = torch.min(pg_loss1, pg_loss2).mean()
                elif args.update_type == "pessimism_only":
                    pg_loss = torch.max(
                        -mb_advantages * ratio,
                        torch.zeros_like(mb_advantages),
                    ).mean()
                elif args.update_type == "optimism_only":
                    pg_loss = torch.min(
                        -mb_advantages * ratio,
                        torch.zeros_like(mb_advantages),
                    ).mean()
                elif args.update_type == "leaky_ppo":
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = (1 - args.leak_coef) * torch.clamp(
                        ratio,
                        1 - clip_coef,
                        1 + clip_coef,
                    ) + args.leak_coef * ratio
                    pg_loss2 = -mb_advantages * pg_loss2
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                elif args.update_type == "rdes":
                    pg_loss = -(mb_advantages * ratio).mean()
                elif args.update_type == "trefree":
                    pg_loss = -torch.clamp(
                        (ratio - 1) * mb_advantages,
                        max=clip_coef,
                    ).mean()
                elif args.update_type == "advantage_pessimism":
                    clipped_ratio = torch.clamp(
                        ratio,
                        1 - clip_coef,
                        1 + clip_coef,
                    )
                    pessimistic_advantage = torch.max(
                        -mb_advantages,
                        torch.zeros_like(mb_advantages),
                    )
                    pg_loss = (pessimistic_advantage * clipped_ratio).mean()
                elif args.update_type == "advantage_optimism":
                    clipped_ratio = torch.clamp(
                        ratio,
                        1 - clip_coef,
                        1 + clip_coef,
                    )
                    optimistic_advantage = torch.min(
                        -mb_advantages,
                        torch.zeros_like(mb_advantages),
                    )
                    pg_loss = (optimistic_advantage * clipped_ratio).mean()
                elif args.update_type == "advantage_neutral":
                    clipped_ratio = torch.clamp(
                        ratio,
                        1 - clip_coef,
                        1 + clip_coef,
                    )
                    pg_loss = (-mb_advantages * clipped_ratio).mean()
                elif args.update_type == "advantage_clipped":
                    clipped_ratio = torch.clamp(
                        ratio,
                        1 - clip_coef,
                        1 + clip_coef,
                    )
                    clipped_advantage = torch.clamp(
                        -mb_advantages,
                        -args.adv_clip_coef,
                        args.adv_clip_coef,
                    )
                    pg_loss = (clipped_advantage * clipped_ratio).mean()
                elif args.update_type == "advantage_pessimism_clipped":
                    clipped_ratio = torch.clamp(
                        ratio,
                        1 - clip_coef,
                        1 + clip_coef,
                    )
                    clipped_advantage = torch.clamp(
                        -mb_advantages,
                        0,
                        args.adv_clip_coef,
                    )
                    pg_loss = (clipped_advantage * clipped_ratio).mean()
                elif args.update_type == "advantage_optimism_clipped":
                    clipped_ratio = torch.clamp(
                        ratio,
                        1 - clip_coef,
                        1 + clip_coef,
                    )
                    clipped_advantage = torch.clamp(
                        -mb_advantages,
                        -args.adv_clip_coef,
                        0,
                    )
                    pg_loss = (clipped_advantage * clipped_ratio).mean()
                elif args.update_type == "advantage_neutral_spma":
                    pg_loss = (-mb_advantages * logratio).mean()
                elif args.update_type == "advantage_clipped_spma":
                    clipped_advantage = torch.clamp(
                        -mb_advantages,
                        -args.adv_clip_coef,
                        args.adv_clip_coef,
                    )
                    pg_loss = (clipped_advantage * logratio).mean()
                elif args.update_type == "advantage_neutral_spma_clipped":
                    clipped_logratio = torch.clamp(
                        logratio,
                        torch.log(torch.tensor(1 - clip_coef)),
                        torch.log(torch.tensor(1 + clip_coef)),
                    )
                    pg_loss = (-mb_advantages * clipped_logratio).mean()
                elif args.update_type == "advantage_clipped_spma_clipped":
                    clipped_advantage = torch.clamp(
                        -mb_advantages,
                        -args.adv_clip_coef,
                        args.adv_clip_coef,
                    )
                    clipped_logratio = torch.clamp(
                        logratio,
                        torch.log(torch.tensor(1 - clip_coef)),
                        torch.log(torch.tensor(1 + clip_coef)),
                    )
                    pg_loss = (clipped_advantage * clipped_logratio).mean()
                elif args.update_type == "advantage_partial_pessimism":
                    clipped_ratio = torch.clamp(
                        ratio,
                        1 - clip_coef,
                        1 + clip_coef,
                    )
                    clipped_advantage = torch.max(
                        -mb_advantages,
                        -args.adv_clip_coef * torch.ones_like(mb_advantages),
                    )
                    pg_loss = (clipped_advantage * clipped_ratio).mean()
                elif args.update_type == "advantage_partial_pessimism_spma":
                    clipped_advantage = torch.max(
                        -mb_advantages,
                        -args.adv_clip_coef * torch.ones_like(mb_advantages),
                    )
                    pg_loss = (clipped_advantage * logratio).mean()
                elif args.update_type == "advantage_partial_pessimism_spma_clipped":
                    clipped_advantage = torch.max(
                        -mb_advantages,
                        -args.adv_clip_coef * torch.ones_like(mb_advantages),
                    )
                    clipped_logratio = torch.clamp(
                        logratio,
                        torch.log(torch.tensor(1 - clip_coef)),
                        torch.log(torch.tensor(1 + clip_coef)),
                    )
                    pg_loss = (clipped_advantage * clipped_logratio).mean()
                else:
                    raise ValueError(
                        "update_type must be either 'ppo' or 'trefree'",
                    )
                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    agent.parameters(),
                    args.max_grad_norm,
                )
                grad_norms.append(grad_norm.sum().item())
                optimizer.step()
                ratios += [ratio.mean().item()]
                ratios_stds += [ratio.std().item()]
                max_ratios += [ratio.max().item()]
                logratios += [logratio.mean().item()]
                logratios_stds += [logratio.std().item()]
                max_logratios += [logratio.max().item()]
                advantages_norm += [
                    (mb_advantages - mb_advantages.mean())
                    / (mb_advantages.std() + 1e-8).mean().item(),
                ]
                max_advantages_norm += [
                    (mb_advantages - mb_advantages.mean())
                    / (mb_advantages.std() + 1e-8).max().item(),
                ]
                advantage_clipfrac += [
                    mb_advantages.abs().gt(args.adv_clip_coef).float().mean().item(),
                ]

                if args.autotune_clip_coef:
                    if np.mean(clipfracs) < 1.5 * args.target_clipfrac:
                        clip_coef -= 0.01
                    elif np.mean(clipfracs) > 1.5 * args.target_clipfrac:
                        clip_coef += 0.01

            if args.target_kl is not None and approx_kl.item() > args.target_kl:
                break
            if args.update_type == "rdes":
                all_ratios = torch.cat(all_ratios, dim=0)
                avg_ratio_deviation = torch.abs(all_ratios - 1.0).mean().item()
                if avg_ratio_deviation > args.clip_coef:
                    break
        # value function updates
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate",
            optimizer.param_groups[0]["lr"],
            global_step,
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/grad_norm", np.mean(grad_norms), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("metrics/ratio", np.mean(ratios), global_step)
        writer.add_scalar("metrics/ratio_std", np.mean(ratios_stds), global_step)
        writer.add_scalar("metrics/max_ratio", np.mean(max_ratios), global_step)
        writer.add_scalar("metrics/logratio", np.mean(logratios), global_step)
        writer.add_scalar("metrics/logratio_std", np.mean(logratios_stds), global_step)
        writer.add_scalar(
            "metrics/max_logratio",
            np.mean(max_logratios),
            global_step,
        )
        writer.add_scalar("metrics/advantage", np.mean(advantages), global_step)
        writer.add_scalar(
            "metrics/advantage_std",
            np.mean(advantages_stds),
            global_step,
        )
        writer.add_scalar("metrics/max_advantage", np.mean(max_advantages), global_step)
        writer.add_scalar(
            "metrics/advantage_norm",
            np.mean(advantages_norm),
            global_step,
        )
        writer.add_scalar(
            "metrics/max_advantage_norm",
            np.mean(max_advantages_norm),
            global_step,
        )
        writer.add_scalar(
            "metrics/advantage_clipfrac",
            np.mean(advantage_clipfrac),
            global_step,
        )
        writer.add_scalar("metrics/clip_coef", clip_coef, global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS",
            int(global_step / (time.time() - start_time)),
            global_step,
        )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "PPO",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()
