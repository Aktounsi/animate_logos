import torch
import numpy as np
import copy
from model_head import MLP
from fitness_function import aesthetic_measure


def init_weights(m):
    for layer in m.hidden:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.00)
    torch.nn.init.xavier_uniform_(m.out.weight)
    m.out.bias.data.fill_(0.00)


def create_random_agents(num_agents, hidden_sizes, out_size):
    agents = []

    for _ in range(num_agents):

        agent = MLP(hidden_sizes, out_size)

        for param in agent.parameters():
            param.requires_grad = False

        init_weights(agent)
        agents.append(agent)

    return agents


# TODO: has to be adapted, depends on modules to transform model output into animated SVG and fitness function that rewards animated SVG
def return_average_reward(X, Y):
  rewards = np.array([aesthetic_measure(X[i], Y[i]) for i in range(X.shape[0])])
  return np.mean(rewards)


def compute_agent_rewards(agents, X):
  agent_rewards = []
  for agent in agents:
    Y_hat = (agent.forward(X) > 0.5).int()
    agent_rewards.append(return_average_reward(X,Y_hat))
  return agent_rewards


def crossover(agents, num_agents, hidden_sizes, out_size):
    children = []
    for _ in range((num_agents - len(agents)) // 2):
        parent1 = np.random.choice(agents)
        parent2 = np.random.choice(agents)
        child1 = MLP(hidden_sizes, out_size)
        child2 = MLP(hidden_sizes, out_size)

        shapes = [param.shape for param in parent1.parameters()]

        genes1_flat = np.concatenate([param.flatten() for param in parent1.parameters()])
        genes2_flat = np.concatenate([param.flatten() for param in parent2.parameters()])

        genes1_unflat = []
        index = 0
        for shape in shapes:
            shape_flat = np.product(shape)
            genes1_unflat.append(genes1_flat[index: (index + shape_flat)].reshape(shape))
            index += shape_flat

        genes2_unflat = []
        index = 0
        for shape in shapes:
            shape_flat = np.product(shape)
            genes2_unflat.append(genes2_flat[index: (index + shape_flat)].reshape(shape))
            index += shape_flat

        for i, param in enumerate(child1.parameters()):
            param = genes1_unflat[i]

        for i, param in enumerate(child2.parameters()):
            param = genes2_unflat[i]

        children.append(child1)
        children.append(child2)
    agents.extend(children)
    return agents


def mutate(agent, mutation_power=0.02):
    child_agent = copy.deepcopy(agent)

    for param in child_agent.parameters():
        if len(param.shape) == 2:
            param += mutation_power * np.random.randn(param.shape[0], param.shape[1])
        if len(param.shape) == 1:
            param += mutation_power * np.random.randn(param.shape[0])

    return child_agent
