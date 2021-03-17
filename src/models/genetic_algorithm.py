import torch, random
import numpy as np
import pandas as pd
import copy
from os import listdir
from os.path import isfile, join

from src.utils import utils
from src.utils import logger

from src.models.animation_prediction import AnimationPredictor
from src.models.surrogate_model import *


def init_weights(m):
    for layer in m.hidden:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.00)
    torch.nn.init.xavier_uniform_(m.out.weight)
    m.out.bias.data.fill_(0.00)


def create_random_agents(num_agents, hidden_sizes, out_size):
    agents = []

    for _ in range(num_agents):

        agent = AnimationPredictor()

        for param in agent.parameters():
            param.requires_grad = False

        # init_weights(agent)
        agents.append(agent)

    return agents


def pad_list(list_):
    return list_ + [[-1] * 13] * (8 - len(list_))


def prepare_surrogate_model_input(model_output, filenames, animation_ids, svg_embeddings):
    model_output_list = [output.tolist() for output in model_output]
    path_level_df = pd.DataFrame(
        {'filename': filenames, 'animation_id': animation_ids, 'model_output': model_output_list})
    concatenated_model_outputs = path_level_df.groupby('filename')['model_output'].apply(list)
    surrogate_model_input = pd.merge(left=concatenated_model_outputs, right=svg_embeddings, on='filename')
    surrogate_model_input['model_output'] = [pad_list(output) for output in surrogate_model_input['model_output']]
    surrogate_model_input[[f'animation_output_{i}' for i in range(8)]] = pd.DataFrame(
        surrogate_model_input['model_output'].tolist(), index=surrogate_model_input.index)
    for i in range(8):
        surrogate_model_input[[f'animation_output_{i}_{j}' for j in range(13)]] = pd.DataFrame(
            surrogate_model_input[f'animation_output_{i}'].tolist(), index=surrogate_model_input.index)
    surrogate_model_input.drop([f'animation_output_{i}' for i in range(8)], inplace=True, axis=1)
    surrogate_model_input.drop('model_output', inplace=True, axis=1)
    return surrogate_model_input


def return_average_reward(model_output, filenames, animation_ids, svg_embeddings):
    rewards = predict_svg_reward(prepare_surrogate_model_input(model_output, filenames, animation_ids, svg_embeddings))
    return np.mean(rewards)


def compute_agent_rewards(agents, X, filenames, animation_ids, svg_embeddings):
    agent_rewards = []
    num_agents = len(agents)
    for i, agent in enumerate(agents):

        model_output = agent(X)
        if i % 20 == 0:
            logger.info(f'Compute rewards for agent {i + 1}/{num_agents}')
        agent_rewards.append(return_average_reward(model_output, filenames, animation_ids, svg_embeddings))
    return agent_rewards


def crossover(agents, num_agents, hidden_sizes, out_size):
    children = []
    for _ in range((num_agents - len(agents)) // 2):
        parent1 = np.random.choice(agents)
        parent2 = np.random.choice(agents)
        child1 = AnimationPredictor()
        child2 = AnimationPredictor()

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
