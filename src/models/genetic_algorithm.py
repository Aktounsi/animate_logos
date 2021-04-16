import torch, random
import numpy as np
import pandas as pd
import copy
from os import listdir
from os.path import isfile, join
from datetime import datetime
from collections import Counter

from src.utils.logger import *

from src.models import config
from src.models.ordinal_classifier_fnn import *
from src.models.animation_prediction import AnimationPredictor
from src.models.train_animation_predictor import *


def init_weights(m):
    # Todo: Update to new model architecture
    for layer in m.hidden:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.00)
    torch.nn.init.xavier_uniform_(m.out.weight)
    m.out.bias.data.fill_(0.00)


def create_random_agents(num_agents):
    agents = []

    for _ in range(num_agents):

        agent = AnimationPredictor(config.a_input_size, config.a_hidden_sizes, config.a_out_sizes)

        for param in agent.parameters():
            param.requires_grad = False

        # init_weights(agent)
        agents.append(agent)

    return agents


def create_animation_vector(animation_prediction):
    # Todo: Maybe put this function to another directory?
    if animation_prediction[0] == 1:
        for i in [8, 9, 10, 11]:
            animation_prediction[i] = -1

    if animation_prediction[1] == 1:
        for i in [6, 7, 9, 10, 11]:
            animation_prediction[i] = -1

    if animation_prediction[2] == 1:
        for i in [6, 7, 8, 10, 11]:
            animation_prediction[i] = -1

    if animation_prediction[3] == 1:
        for i in [6, 7, 8, 9]:
            animation_prediction[i] = -1

    if animation_prediction[4] == 1:
        for i in [6, 7, 8, 9, 10, 11]:
            animation_prediction[i] = -1

    if animation_prediction[5] == 1:
        for i in [6, 7, 8, 9, 10, 11]:
            animation_prediction[i] = -1
    return animation_prediction


def prepare_sm_input(path_vectors, animation_predictions):
    return torch.tensor([list(torch.cat((create_animation_vector(animation_predictions[i]),
                                         path_vectors[i]), 0).detach().numpy())
                         for i in range(path_vectors.shape[0])])


def return_average_reward(path_vectors, animation_predictions):
    # types = list()
    # for i in range(animation_predictions.shape[0]):
    #     types.append(np.argmax(animation_predictions[i][:6].detach().numpy()))
    # info(f'Type distribution: {Counter(types)}')
    rewards = predict(prepare_sm_input(path_vectors, animation_predictions))
    return np.mean(rewards)


def compute_agent_rewards(agents, path_vectors, verbose=5):
    steps = len(agents) // verbose
    agent_rewards = []
    num_agents = len(agents)
    for i, agent in enumerate(agents):
        animation_predictions = agent(path_vectors)
        avg_rewards = return_average_reward(path_vectors, animation_predictions)
        if i % steps == 0:
            info(f'Computed avg rewards for agent {i + 1}/{num_agents}: {avg_rewards}')
        agent_rewards.append(avg_rewards)
    return agent_rewards


def crossover(agents, num_agents):
    children = []
    for _ in range((num_agents - len(agents)) // 2):
        parent1 = np.random.choice(agents)
        parent2 = np.random.choice(agents)
        child1 = AnimationPredictor(config.a_input_size, config.a_hidden_sizes, config.a_out_sizes)
        child2 = AnimationPredictor(config.a_input_size, config.a_hidden_sizes, config.a_out_sizes)

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
