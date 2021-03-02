import torch
from datetime import datetime
from src.models.genetic_algorithm import *


def train_model_head(svg_dataset, hidden_sizes=[256, 192], out_size=21,
                     num_agents=100, top_parent_limit=10, generations=10):

    # Create model input from svg_dataset
    filenames = svg_dataset.pop('filename')
    animation_ids = svg_dataset.pop('animation_id')
    model_input = torch.tensor(svg_dataset.to_numpy())

    # disable gradients as we will not use them
    torch.set_grad_enabled(False)

    # initialize number of agents
    overall_start = datetime.now()
    agents = create_random_agents(num_agents=num_agents, hidden_sizes=hidden_sizes, out_size=out_size)

    logger.info('Model summary')
    print('=' * 100)
    print(f'Number of training instances: {len(svg_dataset)}')
    print(f'Number of agents: {num_agents}')
    print(f'Top parent limit: {top_parent_limit}')
    print(f'Number of generations: {generations}')
    print(f'Hidden sizes: {hidden_sizes}')
    print(f'Output size: {out_size}')
    print('=' * 100)

    for generation in range(generations):
        start = datetime.now()
        rewards = compute_agent_rewards(agents=agents, X=model_input,
                                        filenames=filenames,
                                        animation_ids=animation_ids)
        sorted_parent_indexes = np.argsort(rewards)[::-1][:top_parent_limit].astype(int)
        logger.info(f'Results for generation {generation + 1}/{generations}')
        print('=' * 100)
        top_agents = [agents[best_parent] for best_parent in sorted_parent_indexes]
        top_rewards = [rewards[best_parent] for best_parent in sorted_parent_indexes]

        print(f'Generation {generation} '
              f'| Mean rewards: {np.mean(rewards)} | Mean of top 10: {np.mean(top_rewards[:10])}')
        print(f'Top {top_parent_limit} agents: {sorted_parent_indexes}')
        print(f'Rewards for top {top_parent_limit} agents: {top_rewards}')
        children_agents = crossover(agents=top_agents, num_agents=num_agents,
                                    hidden_sizes=hidden_sizes, out_size=out_size)
        children_agents = [mutate(agent) for agent in children_agents]
        agents = children_agents
        stop = datetime.now()
        print('=' * 100)
        logger.info(f'Operation time for generation {generation}: {stop - start}')

    overall_stop = datetime.now()
    logger.info(f'Overall operation time: {overall_stop - overall_start}')
