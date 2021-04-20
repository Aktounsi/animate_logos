import os, sys, torch, pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from argparse import ArgumentParser

from src.models import config
from src.utils.logger import *
from src.models.genetic_algorithm import *
from src.features.get_svg_size_pos import get_relative_pos_to_bounding_box_of_animated_paths


def retrieve_m1_predictions(input_data):
    m1_path_vectors = torch.tensor(input_data[config.m1_features].to_numpy(), dtype=torch.float)
    m1 = pickle.load(open(config.m1_path, 'rb'))
    path_predictions = m1.predict(m1_path_vectors)
    input_data['animated'] = path_predictions

    info(f'Number of animated paths: {sum(path_predictions)}/{len(path_predictions)}')
    return input_data


def retrieve_animation_midpoints(input_data, drop=True):
    # Integrate midpoint of animation as feature
    animated_input_data = input_data[input_data['animated'] == 1]
    gb = animated_input_data.groupby('filename')['animation_id'].apply(list)

    input_data = pd.merge(left=input_data, right=gb, on='filename')
    input_data.rename(columns={'animation_id_x': 'animation_id', 'animation_id_y': 'animated_animation_ids'},
                      inplace=True)

    if drop:
        info("Not animated paths won't be used for training")
        input_data = input_data[input_data['animated'] == 1]
    else:
        info("Not animated paths will be used for training")
    input_data.reset_index(drop=True, inplace=True)
    info('Start extraction midpoint of animated paths as feature')
    input_data["rel_position_to_animations"] = input_data.apply(
        lambda row: get_relative_pos_to_bounding_box_of_animated_paths(f"data/initial_svgs/{row['filename']}.svg",
                                                                       int(row["animation_id"]),
                                                                       row["animated_animation_ids"]), axis=1)
    input_data["rel_x_position_to_animations"] = input_data["rel_position_to_animations"].apply(lambda row: row[0])
    input_data["rel_y_position_to_animations"] = input_data["rel_position_to_animations"].apply(lambda row: row[1])
    return input_data


def train_animation_predictor(path_vectors, hidden_sizes=config.a_hidden_sizes, out_size=config.a_out_sizes,
                     num_agents=100, top_parent_limit=10, generations=10, timestamp=''):
    # disable gradients as we will not use them
    torch.set_grad_enabled(False)

    # initialize number of agents
    overall_start = datetime.now()
    agents = create_random_agents(num_agents=num_agents)

    info('AP model summary')
    print('=' * 100)
    print(f'Number of training instances: {len(path_vectors)}')
    print(f'Number of agents: {num_agents}')
    print(f'Top parent limit: {top_parent_limit}')
    print(f'Number of generations: {generations}')
    print(f'Hidden sizes: {hidden_sizes}')
    print(f'Output size: {out_size}')
    print('=' * 100)

    animation_predictions = pd.DataFrame(
        {'generation': [], 'agent_rank': [], 'agent_mean_reward': [], 'translate': [], 'scale': [],
         'rotate': [], 'skew': [], 'fill': [], 'opacity': []})
    parameters = pd.DataFrame({'generation': [], 'child': [], 'before': [], 'after': []})

    for generation in range(generations):
        start = datetime.now()
        info(f'Generation {generation + 1}/{generations}')
        rewards, predictions = compute_agent_rewards(agents=agents, path_vectors=path_vectors)
        sorted_parent_indexes = np.argsort(rewards)[::-1][:top_parent_limit].astype(int)
        top_agents = [agents[best_parent] for best_parent in sorted_parent_indexes]
        top_rewards = [rewards[best_parent] for best_parent in sorted_parent_indexes]
        top_predictions = [predictions[best_parent] for best_parent in sorted_parent_indexes]

        for i, agent in enumerate(top_agents):
            types = list()
            for j in range(top_predictions[i].shape[0]):
                types.append(np.argmax(top_predictions[i][j][:6].detach().numpy()))
            type_counts = Counter(types)
            animation_dict = {'generation': generation, 'agent_rank': i, 'agent_mean_reward': top_rewards[i],
                              'translate': type_counts[0], 'scale': type_counts[1], 'rotate': type_counts[2],
                              'skew': type_counts[3], 'fill': type_counts[4], 'opacity': type_counts[5]}
            animation_predictions = animation_predictions.append(animation_dict, ignore_index=True)

        children_agents, before, after = crossover(agents=top_agents, num_agents=num_agents)

        for child in range(len(before)):
            params_dict = {'generation': generation, 'child': child, 'before': before[child], 'after': after[child]}
            parameters = parameters.append(params_dict, ignore_index=True)

        children_agents = [mutate(agent) for agent in children_agents]
        agents = children_agents

        stop = datetime.now()
        info(f'Operation time: {stop - start}')

        print('-' * 100)

        print(f'Mean rewards: {np.mean(rewards)} | Mean of top 10: {np.mean(top_rewards[:10])}')
        print(f'Top {top_parent_limit} agents: {sorted_parent_indexes}')
        print(f'Rewards for top {top_parent_limit} agents: {top_rewards[:top_parent_limit]}')

        print('=' * 100)

    overall_stop = datetime.now()
    info(f'Overall operation time: {overall_stop - overall_start}')

    animation_predictions.to_csv(f'logs/{timestamp}_animation_predictions.csv', index=False)
    parameters.to_csv(f'logs/{timestamp}_parameters.csv', index=False)

    return top_agents[0]


def main(data_path='data/model_1/model_1_train.csv', drop=True,
         num_agents=100, top_parent_limit=20, generations=50, timestamp=''):
    info(f'Data source: {data_path}')
    input_data = pd.read_csv(data_path)

    # Apply model one to get predictions whether to animate paths or not
    input_data = retrieve_m1_predictions(input_data)

    # Retrieve features describing the midpoint of animated paths
    input_data = retrieve_animation_midpoints(input_data, drop=drop)

    # Prepare path vectors for animation prediction
    path_vectors = torch.tensor(input_data[config.sm_features].to_numpy(), dtype=torch.float)

    # Perform genetic algorithm for animation prediction
    top_model = train_animation_predictor(path_vectors=path_vectors, num_agents=num_agents,
                                          top_parent_limit=top_parent_limit, generations=generations,
                                          timestamp=timestamp)

    # Save best model for animation prediction
    torch.save(top_model, 'models/ap_best_model.pkl')
    torch.save(top_model.state_dict(), 'models/ap_best_model_state_dict.pth')

    return top_model


if __name__ == '__main__':
    os.chdir('../..')

    ap = ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-s", "--save", required=False, action='store_true', help="if set, output will be saved to file")
    ap.add_argument("-d", "--drop", required=False, action='store_false', help="if set, not animated paths "
                                                                               "will be kept for training")
    ap.add_argument("-a", "--n_agents", required=False, default=100, help="number of agents")
    ap.add_argument("-t", "--top_parents", required=False, default=20, help="number of top agents to be considered, "
                                                                            "should be even number")
    ap.add_argument("-g", "--generations", required=False, default=50, help="number of generations")
    args = vars(ap.parse_args())

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    if args['save']:
        Path('logs/').mkdir(parents=True, exist_ok=True)
        log_file = f"logs/{timestamp}_ap_training.txt"
        info(f'Saving enabled, log file: {log_file}')
        sys.stdout = open(log_file, 'w')
        main(num_agents=int(args['n_agents']), top_parent_limit=int(args['top_parents']),
             generations=int(args['generations']), drop=args['drop'], timestamp=timestamp)
        sys.stdout.close()
    else:
        main(num_agents=int(args['n_agents']), top_parent_limit=int(args['top_parents']),
             generations=int(args['generations']), drop=args['drop'], timestamp=timestamp)
