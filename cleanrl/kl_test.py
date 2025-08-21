import numpy as np
import torch
from torch import nn


def squared_kl(p, q):
    return (torch.log(p) - torch.log(q)) ** 2 / 2


def log_ratio_kl(p, q):
    return torch.log(p) - torch.log(q)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = layer_init(nn.Linear(8, 5), std=0.01)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        # probs = Categorical(logits=logits)
        # if action is None:
        # action = probs.sample()
        return action, logits, logits
        # return action, probs.log_prob(action), probs.entropy()


if __name__ == "__main__":
    actor = Agent()  # Replace `envs=None` with actual environment if needed
    actor2 = Agent()  # Another instance for comparison
    x = torch.randn(4, 8)  # Example input
    action, log_prob, entropy = actor.get_action_and_value(x)
    print(log_prob)
    # action2, log_prob2, entropy2 = actor2.get_action_and_value(x)
    log_prob2 = torch.randn_like(
        log_prob,
    )  # Simulating another log probability for testing
    squared_kl_val = squared_kl(log_prob.exp(), log_prob2.exp())
    grads = torch.autograd.grad(
        squared_kl_val.sum(),
        actor.parameters(),
    )
    print(grads)
    action, log_prob, entropy = actor.get_action_and_value(x)
    print(log_prob)
    squared_kl_forward = squared_kl(log_prob2.exp(), log_prob.exp())
    grads_forward = torch.autograd.grad(
        squared_kl_forward.sum(),
        actor.parameters(),
    )
    print(grads_forward)
    print((grads[1] - grads_forward[1]).abs().max())
    action, log_prob, entropy = actor.get_action_and_value(x)
    log_ratio_kl_val = log_ratio_kl(log_prob.exp(), log_prob2.exp())
    grads_log_ratio = torch.autograd.grad(
        log_ratio_kl_val.sum(),
        actor.parameters(),
    )
    print(grads_log_ratio)
    action, log_prob, entropy = actor.get_action_and_value(x)
    log_ratio_kl_forward = log_ratio_kl(log_prob2.exp(), log_prob.exp())
    grads_log_ratio_forward = torch.autograd.grad(
        log_ratio_kl_forward.sum(),
        actor.parameters(),
    )
    print(grads_log_ratio_forward)
    print((grads_log_ratio[1] - grads_log_ratio_forward[1]).abs().max())
