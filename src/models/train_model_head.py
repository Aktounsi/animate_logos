from genetic_algorithm import *
from transform_model_output_to_animation_states import *
from model_head import *
from fitness_function import *


# TODO: DOES NOT WORK YET!
def train_model_head(svg_dataset, hidden_sizes = [256], num_agents = 100, top_parent_limit = 10, generations = 10):

    # disable gradients as we will not use them
    torch.set_grad_enabled(False)

    # initialize number of agents
    num_agents = num_agents
    agents = create_random_agents(num_agents)

    # select how many top agents are supposed to be considered as agents
    top_parent_limit = 10

    # select how many generations evolution is supposed to run
    generations = 10

    for generation in range(generations):
      rewards = compute_agent_rewards(agents, inp)
      sorted_parent_indexes = np.argsort(rewards)[::-1][:top_parent_limit].astype(int)
      print("")
      print("")
      top_agents = [agents[best_parent] for best_parent in sorted_parent_indexes]
      top_rewards = [rewards[best_parent] for best_parent in sorted_parent_indexes]
      print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 10: ", np.mean(top_rewards[:10]))
      print("Top ",top_parent_limit," agents", sorted_parent_indexes)
      print("Rewards for top", top_parent_limit," agents: ",top_rewards)
      children_agents = crossover(top_agents, num_agents)
      children_agents = [mutate(agent) for agent in children_agents]
      agents = children_agents