import torch
import torch.nn as nn
import numpy as np
import copy
from fitness_function import aesthetic_measure
from transform_model_output_to_animation_states import interpolate_svg

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(30, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 16, bias=True),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.fc(inputs)
        return x


def init_weights(m):
    for layer in m.fc:

        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.00)


def create_random_agents(num_agents):
    agents = []

    for _ in range(num_agents):

        agent = MLP()

        for param in agent.parameters():
            param.requires_grad = False

        init_weights(agent)
        agents.append(agent)

    return agents


def compute_agent_rewards(driver, agents, inp_model, inp_ff):
    agent_rewards = []
    for agent in agents:
        print("New Agent")
        animation = agent.forward(inp_model) > 0.5
        agent_rewards.append(return_average_reward(driver, inp_model, inp_ff, animation))

    return agent_rewards


def return_average_reward(inp_model, inp_ff, animation):
    rewards = []
    for i in range(inp_model.shape[0]):
        interpolate_svg(logo, total_duration, steps, element_id, type, begin, dur, from_, to, fromY=None, toY=None, repeatCount=1, fill='freeze')
        reward = aesthetic_measure(path_file_animated)
        rewards.append(reward)
    return np.mean(rewards)


def crossover(agents, num_agents):
    children = []
    for _ in range((num_agents - len(agents)) // 2):
        parent1 = np.random.choice(agents)
        parent2 = np.random.choice(agents)
        child1 = MLP()
        child2 = MLP()

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
