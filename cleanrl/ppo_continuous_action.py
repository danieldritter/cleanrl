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
    """the surrogate clipping coefficient"""
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
    use_kl_penalty: bool = False
    """whether to use KL penalty instead of clipping"""
    kl_penalty_coef: float = 1.0
    """the coefficient of the KL penalty"""
    kl_direction: str = "forward"
    mdpo_anneal_kl_penalty: bool = False
    """whether to use KL penalty with MDPO annealing"""
    anneal_kl_penalty: bool = False
    """whether to use KL penalty with linear annealing"""
    adaptive_kl_penalty: bool = False
    """whether to use adaptive KL penalty"""
    kl_penalty_target: float = 0.01
    kl_lower_bound: float = 0.0
    """Loss is not penalized if KL is below this value"""
    kl_estimator: str = "low_var"
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
    loss_name: str = "ppo"  # one of "mdpo", "ppo", "klppo", "zeroeta"
    sparsity_steps: int = 1
    """number of steps to skip before giving reward. -1 holds reward till end of episode"""
    is_bandit: bool = False
    """Whether to run the environment as a bandit problem (rewards only returned at end of episode, total reward used as reward for all states)"""
    is_discretized: bool = False
    """Whether to use discretized actions"""
    num_bins: int = 7
    """number of bins for discretized actions, only used if is_discretized=True"""


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
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
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


class DiscretizedAgent(nn.Module):
    def __init__(self, envs, orthogonal=True, num_bins=7):
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
        num_actions = np.prod(envs.single_action_space.shape)
        self.envs = envs
        self.num_bins = num_bins
        self.all_actions = []
        for i in range(num_actions):
            low = envs.single_action_space.low[i]
            high = envs.single_action_space.high[i]
            actions = self.get_discretized_actions(low, high, num_bins)
            self.all_actions.append(actions)
        self.all_actions = torch.tensor(self.all_actions).to(
            device=next(self.parameters()).device,
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
                orthogonal=orthogonal,
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64), orthogonal=orthogonal),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, np.prod(envs.single_action_space.shape) * num_bins),
                std=0.01,
                orthogonal=orthogonal,
            ),
        )

    def get_discretized_actions(self, low, high, num_bins):
        actions = []
        for i in range(num_bins):
            action = low + i * (high - low) / (num_bins - 1)
            actions.append(action)
        return actions

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, action=None):
        action_logits = self.actor(x)
        action_logits = action_logits.view(
            -1,
            np.prod(self.envs.single_action_space.shape),
            self.num_bins,
        )
        dist = torch.distributions.multinomial.Multinomial(
            logits=action_logits,
            total_count=1,
        )
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        action_indices = action.argmax(dim=-1)  # (batch_size, num_actions)
        entropy = dist.entropy()

        # Gather continuous action values
        # Expand all_actions to have batch dimension
        all_actions_expanded = self.all_actions.unsqueeze(0).expand(
            action_indices.shape[0],
            -1,
            -1,
        )  # (batch_size, num_actions, num_bins)

        # Gather the continuous values
        action_continuous = torch.gather(
            all_actions_expanded,
            dim=2,
            index=action_indices.unsqueeze(2),
        ).squeeze(2)  # (batch_size, num_actions)
        return (
            action_continuous,
            log_prob.sum(dim=-1),
            entropy.sum(dim=-1),
            torch.zeros_like(action_continuous),
            torch.zeros_like(action_continuous),
        )

    def convert_action_to_onehot(self, action):
        # converts output of get_action back to the one-hot
        # sampled from the dist in that function

        # action has shape (batch_size, num_actions) with continuous values
        batch_size, num_actions = action.shape

        # Initialize one-hot tensor with shape (batch_size, num_actions, num_bins)
        action_onehot = torch.zeros(
            batch_size,
            num_actions,
            self.num_bins,
            device=action.device,
        )

        # For each action dimension, find which bin the continuous value corresponds to
        for i in range(num_actions):
            # Get the discrete bin values for this action dimension
            bin_values = self.all_actions[i]  # shape (num_bins,)

            # Find closest bin for each batch element
            # action[:, i] has shape (batch_size,)
            # bin_values has shape (num_bins,)
            # Compute distances between each action value and all bin values
            diffs = torch.abs(
                action[:, i, None] - bin_values[None, :],
            )  # (batch_size, num_bins)
            bin_indices = torch.argmin(diffs, dim=1)  # (batch_size,)

            # Set one-hot encoding
            action_onehot[torch.arange(batch_size), i, bin_indices] = 1.0

        return action_onehot

    def get_action_and_value(self, x, action=None):
        action_logits = self.actor(x)
        action_logits = action_logits.view(
            -1,
            np.prod(self.envs.single_action_space.shape),
            self.num_bins,
        )
        dist = torch.distributions.multinomial.Multinomial(
            logits=action_logits,
            total_count=1,
        )
        if action is None:
            action = dist.sample()
        else:
            action = self.convert_action_to_onehot(action)

        log_prob = dist.log_prob(action)
        action_indices = action.argmax(dim=-1)  # (batch_size, num_actions)
        entropy = dist.entropy()

        # Gather continuous action values
        # Expand all_actions to have batch dimension
        all_actions_expanded = self.all_actions.unsqueeze(0).expand(
            action_indices.shape[0],
            -1,
            -1,
        )  # (batch_size, num_actions, num_bins)

        # Gather the continuous values
        action_continuous = torch.gather(
            all_actions_expanded,
            dim=2,
            index=action_indices.unsqueeze(2),
        ).squeeze(2)  # (batch_size, num_actions)
        return (
            action_continuous,
            log_prob.sum(dim=-1),
            entropy.sum(dim=-1),
            self.critic(x),
            torch.zeros_like(action_continuous),
            torch.zeros_like(action_continuous),
        )


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
    if args.use_kl_penalty:
        if args.kl_penalty_coef == 0.0:
            args.loss_name = "zeroeta"
        elif args.kl_direction == "forward":
            args.loss_name = "klppo"
        elif args.kl_direction == "reverse":
            args.loss_name = "mdpo"
    else:
        args.loss_name = "ppo"
    run_name = args.exp_name
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
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
                args.normalize_reward,
                args.normalize_obs,
            )
            for i in range(args.num_envs)
        ],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )
    if not args.is_discretized:
        agent = Agent(envs, orthogonal=args.init_weight_orthogonal).to(device)
    else:
        agent = DiscretizedAgent(
            envs,
            orthogonal=args.init_weight_orthogonal,
            num_bins=args.num_bins,
        ).to(device)
        assert args.kl_estimator != "closed_form_gaussian", (
            "closed_form_gaussian kl estimator is not supported for discretized actions",
        )
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
    if args.use_kl_penalty:
        kl_penalty_coef = args.kl_penalty_coef
        if args.adaptive_kl_penalty:
            assert not args.mdpo_anneal_kl_penalty, (
                "adaptive_kl_penalty and mdpo_anneal_kl_penalty cannot be used together"
            )
            assert not args.anneal_kl_penalty, (
                "adaptive_kl_penalty and anneal_kl_penalty cannot be used together"
            )
        if args.mdpo_anneal_kl_penalty:
            assert not args.adaptive_kl_penalty, (
                "adaptive_kl_penalty and mdpo_anneal_kl_penalty cannot be used together"
            )
            assert not args.anneal_kl_penalty, (
                "mdpo_anneal_kl_penalty and anneal_kl_penalty cannot be used together"
            )

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        if args.use_kl_penalty and args.mdpo_anneal_kl_penalty:
            kl_penalty_coef = args.kl_penalty_coef / (
                1 - (iteration - 1) / args.num_iterations
            )
        if args.use_kl_penalty and args.anneal_kl_penalty:
            kl_penalty_coef = args.kl_penalty_coef * (
                1 - (iteration - 1) / args.num_iterations
            )
        reward_sum = 0
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
            if not args.is_bandit:
                next_done = np.logical_or(terminations, truncations)
                if args.sparsity_steps != 1:
                    reward_sum += reward
                if args.sparsity_steps == -1 and "final_info" in infos:
                    rewards[step] = torch.tensor(reward_sum).to(device).view(-1)
                    reward_sum = 0
                if args.sparsity_steps > 1 and (
                    step % args.sparsity_steps == 0 or "final_info" in infos
                ):
                    rewards[step] = torch.tensor(reward_sum).to(device).view(-1)
                    reward_sum = 0
                if args.sparsity_steps == 1:
                    rewards[step] = torch.tensor(reward).to(device).view(-1)
            else:
                next_done = np.logical_or(terminations, truncations)
                reward_sum += reward
                if "final_info" in infos:
                    rewards[:step] = torch.tensor(reward_sum).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}",
                        )
                        writer.add_scalar(
                            "charts/episodic_return",
                            info["episode"]["r"],
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/episodic_length",
                            info["episode"]["l"],
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
        for epoch in range(args.update_epochs):
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
                if args.use_kl_penalty:
                    with torch.no_grad():
                        # forward KLs
                        mc_forward_kl = (-logratio).mean()
                        low_var_forward_kl = compute_low_var_kl(
                            b_logprobs[mb_inds],
                            newlogprob,
                        ).mean()
                        closed_form_forward_kl = closed_form_gaussian_kl(
                            b_action_means[mb_inds],
                            b_action_logstds[mb_inds],
                            new_action_mean,
                            new_action_logstd,
                        ).mean()
                        # reverse KLs
                        mc_reverse_kl = (ratio * logratio).mean()
                        low_var_reverse_kl = (
                            ratio
                            * compute_low_var_kl(
                                newlogprob,
                                b_logprobs[mb_inds],
                            )
                        ).mean()
                        closed_form_reverse_kl = (
                            ratio
                            * closed_form_gaussian_kl(
                                new_action_mean,
                                new_action_logstd,
                                b_action_means[mb_inds],
                                b_action_logstds[mb_inds],
                            )
                        ).mean()

                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean(),
                        ]
                    if args.kl_direction == "forward":
                        if args.kl_estimator == "low_var":
                            approx_kl_full = compute_low_var_kl(
                                b_logprobs[mb_inds],
                                newlogprob,
                            )
                        elif args.kl_estimator == "standard":
                            approx_kl_full = -logratio
                        elif args.kl_estimator == "closed_form_gaussian":
                            approx_kl_full = closed_form_gaussian_kl(
                                b_action_means[mb_inds],
                                b_action_logstds[mb_inds],
                                new_action_mean,
                                new_action_logstd,
                            )
                    elif args.kl_direction == "reverse":
                        if args.kl_estimator == "low_var":
                            approx_kl_full = ratio * compute_low_var_kl(
                                newlogprob,
                                b_logprobs[mb_inds],
                            )
                        elif args.kl_estimator == "standard":
                            approx_kl_full = ratio * logratio
                        elif args.kl_estimator == "closed_form_gaussian":
                            approx_kl_full = ratio * closed_form_gaussian_kl(
                                new_action_mean,
                                new_action_logstd,
                                b_action_means[mb_inds],
                                b_action_logstds[mb_inds],
                            )
                    else:
                        raise ValueError(
                            "kl_direction must be either 'forward' or 'reverse'",
                        )
                    approx_kl = approx_kl_full.mean()

                else:
                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef)
                            .float()
                            .mean()
                            .item(),
                        ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                if args.use_kl_penalty:
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss = (
                        pg_loss1
                        + kl_penalty_coef
                        * torch.max(
                            torch.zeros_like(approx_kl),
                            approx_kl - args.kl_lower_bound,
                        )
                    ).mean()
                    with torch.no_grad():
                        adv_kl_ratio = (pg_loss1 / (kl_penalty_coef * approx_kl)).mean()
                        adv_kl_ratio_no_coef = (pg_loss1 / approx_kl).mean()
                        adv_kl_ratio_sum = (
                            pg_loss1.sum() / (kl_penalty_coef * approx_kl).sum()
                        )
                    adv_kl_ratio_sum_no_coef = pg_loss1.sum() / approx_kl.sum()
                else:
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio,
                        1 - args.clip_coef,
                        1 + args.clip_coef,
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    with torch.no_grad():
                        adv_kl_ratio_no_coef = (pg_loss1 / approx_kl).mean()
                        adv_kl_ratio_sum_no_coef = pg_loss1.sum() / approx_kl.sum()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
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
            if args.target_kl is not None and approx_kl.item() > args.target_kl:
                break
        if args.use_kl_penalty and args.adaptive_kl_penalty:
            kl_error = np.clip(
                (approx_kl.item() - args.kl_penalty_target) / args.kl_penalty_target,
                -0.2,
                0.2,
            )
            # 0.1 here is from original paper, verl uses a horizon schedule curr_step/T instead
            kl_penalty_coef = kl_penalty_coef * (1 + kl_error)
        # value function updates
        vf_grad_norms = []

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate",
            optimizer.param_groups[0]["lr"],
            global_step,
        )
        if args.use_kl_penalty:
            writer.add_scalar("charts/kl_penalty_coef", kl_penalty_coef, global_step)
            writer.add_scalar("charts/adv_kl_ratio", adv_kl_ratio, global_step)
            writer.add_scalar("charts/adv_kl_ratio_sum", adv_kl_ratio_sum, global_step)
        writer.add_scalar(
            "charts/adv_kl_ratio_no_coef",
            adv_kl_ratio_no_coef.item(),
            global_step,
        )
        writer.add_scalar(
            "charts/adv_kl_ratio_sum_no_coef",
            adv_kl_ratio_sum_no_coef.item(),
            global_step,
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar(
            "losses/mc_forward_kl",
            mc_forward_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "losses/low_var_forward_kl",
            low_var_forward_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "losses/closed_form_forward_kl",
            closed_form_forward_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "losses/mc_reverse_kl",
            mc_reverse_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "losses/low_var_reverse_kl",
            low_var_reverse_kl.item(),
            global_step,
        )
        writer.add_scalar(
            "losses/closed_form_reverse_kl",
            closed_form_reverse_kl.item(),
            global_step,
        )
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/grad_norm", np.mean(grad_norms), global_step)
        writer.add_scalar("losses/vf_grad_norm", np.mean(vf_grad_norms), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
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
