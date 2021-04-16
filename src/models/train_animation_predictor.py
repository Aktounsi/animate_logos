import sys, torch, pickle
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import Counter

from src.models import config
from src.utils.logger import *
from src.models.genetic_algorithm import *
from src.features.get_svg_size_pos import get_relative_pos_to_bounding_box_of_animated_paths


def train_animation_predictor(path_vectors, hidden_sizes=config.a_hidden_sizes, out_size=config.a_out_sizes,
                     num_agents=100, top_parent_limit=10, generations=10):
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

    for generation in range(generations):
        start = datetime.now()
        info(f'Generation {generation + 1}/{generations}')
        rewards = compute_agent_rewards(agents=agents, path_vectors=path_vectors)
        sorted_parent_indexes = np.argsort(rewards)[::-1][:top_parent_limit].astype(int)
        top_agents = [agents[best_parent] for best_parent in sorted_parent_indexes]
        top_rewards = [rewards[best_parent] for best_parent in sorted_parent_indexes]

        top_predictions = top_agents[0](path_vectors)
        types = list()
        for i in range(top_predictions.shape[0]):
            types.append(np.argmax(top_predictions[i][:6].detach().numpy()))
        info(f'Type distribution best agent: {Counter(types)}')

        children_agents = crossover(agents=top_agents, num_agents=num_agents)
        children_agents = [mutate(agent) for agent in children_agents]
        agents = children_agents

        stop = datetime.now()
        info(f'Operation time: {stop - start}')

        print('-' * 100)

        print(f'Mean rewards: {np.mean(rewards)} | Mean of top 10: {np.mean(top_rewards[:10])}')
        print(f'Top {top_parent_limit} agents: {sorted_parent_indexes}')
        print(f'Rewards for top {top_parent_limit} agents: {top_rewards}')

        print('=' * 100)

    overall_stop = datetime.now()
    info(f'Overall operation time: {overall_stop - overall_start}')
    return top_agents[0]


def main(data_path='../../data/model_1/model_1_train.csv'):
    info(f'Data source: {data_path}')
    input_data = pd.read_csv(data_path)
    drop_columns = ['filename', 'animation_id',
                    'stroke_width', 'opacity', 'stroke_opacity', 'stroke_r', 'stroke_g', 'stroke_b', 'svg_stroke_r',
                    'diff_stroke_r', 'svg_stroke_g', 'diff_stroke_g', 'svg_stroke_b', 'diff_stroke_b', 'href']
    input_data.drop(drop_columns, axis=1, inplace=True)

    # Change order of input data
    col_order = ['']

    # Create model input
    model_one_path_vectors = torch.tensor(input_data.to_numpy(), dtype=torch.float)
    model_one = pickle.load(open('../../models/model_1_extra_trees_classifier.sav', 'rb'))
    path_predictions = model_one.predict(model_one_path_vectors)

    info(f'Number of animated paths: {sum(path_predictions)}/{len(path_predictions)}')

    # Todo: Integrate Tim's features
    input_data['add_feature_1'] = [random.uniform(0, 1) for _ in range(len(input_data))]
    input_data['add_feature_2'] = [random.uniform(0, 1) for _ in range(len(input_data))]
    sm_path_vectors = torch.tensor(input_data.to_numpy(), dtype=torch.float)
    sm_path_vectors = torch.tensor([list(sm_path_vectors[i])
                                    for i in range(sm_path_vectors.shape[0]) if path_predictions[i] == 1])

    top_model = train_animation_predictor(path_vectors=sm_path_vectors,
                                          num_agents=100, top_parent_limit=20, generations=10)
    torch.save(top_model, '../../models/ap_best_model.pkl')
    torch.save(top_model.state_dict(), '../../models/ap_best_model_state_dict.pth')
    return top_model


if __name__ == '__main__':
    if 'save' in sys.argv:
        Path('../../logs/').mkdir(parents=True, exist_ok=True)
        log_file = f"../../logs/{datetime.now().strftime('%Y%m%d_%H%M_ap_training.txt')}"
        info(f'Saving enabled, log file: {log_file}')
        sys.stdout = open(log_file, 'w')
        main()
        sys.stdout.close()
    else:
        main()
