import torch
import numpy as np
import pandas as pd
import copy
from os import listdir
from os.path import isfile, join

from src.utils import utils
from src.utils import logger

from src.models.animation_prediction import AnimationPredictor
from src.models.fitness_function import aesthetic_measure
from src.models.transform_model_output_to_animation_states import interpolate_svg, convert_svgs_in_folder


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


def return_reward(path_output, filename, animation_id, idx):
    if idx % 50 == 0:
        logger.info(f'Current path index: {idx}')
    try:
        # Interpolate animation of the current path in its corresponding SVG file and convert interpolations to PNG
        interpolate_svg(logo=f'./data/svgs/{filename}.svg',
                        total_duration=5,
                        steps=10,
                        animation_id=animation_id,
                        output=path_output)
        convert_svgs_in_folder('./data/interpolated_logos')

        # Compute aesthetic measure for the interpolated animation
        interpolated_files = [join('./data/interpolated_logos', f) for f in listdir('./data/interpolated_logos') if
                              isfile(join('./data/interpolated_logos', f))]
        interpolated_files.sort()
        reward = aesthetic_measure(interpolated_files)
        utils.delete_dir(['./data/interpolated_logos'])
        return reward
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(f'Failed computing reward for file={filename} and animation id={animation_id}; '
                     f'Return reward of -1')
        logger.error(f'Raised exception: {e}')
        utils.delete_dir(['./data/interpolated_logos'])
        return -1


def pad_list(l):
    return l + [[-1] * 13] * (8 - len(l))


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
    rewards = predict(prepare_surrogate_model_input(model_output, filenames, animation_ids, svg_embeddings))
    return np.mean(rewards)


def compute_agent_rewards(agents, X, filenames, animation_ids, svg_embeddings):
    agent_rewards = []
    num_agents = len(agents)
    for i, agent in enumerate(agents):
        model_output = agent(X)
        # if i % 10 == 0:
        logger.info(f'Compute rewards for agent {i + 1}/{num_agents}')
        agent_rewards.append(return_average_reward(model_output, filenames, animation_ids, svg_embeddings))
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
