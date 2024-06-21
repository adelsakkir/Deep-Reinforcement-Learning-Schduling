import copy
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.neural_network import MLPRegressor
import pickle

from agents import RandomAgent, GreedyAgent1, DQNAgent_online, DQNAgent_offline
from environment import SchedulingEnvironment
from tools import build_offline_dataset, compute_td_target, visualize_loss

#############################################
# Definitions in experiments.py: functions that execute the episodes using the three different agents
#
# 1. Greedy agent algorithms
# 2. Online DQN agent algorithms
# 3. Offline DQN agent algorithms
#############################################

# Greedy agent algorithms
def greedy1(job_list, machine_list, show_schedule =False):

    # print(job_list[2].id)

    env = SchedulingEnvironment(job_list, machine_list)
    agent = GreedyAgent1()
    obs = env.reset()
    total_reward = 0
    current_state = env.reset()
    done = False
    t=0
    while not done:
        action = agent.select_action(obs)
        next_state, reward, done = env.step(action)
        env.current_state =next_state
        total_reward += reward
        obs = next_state
        #obs.display_state() # uncomment for debugging
        if done:
            if show_schedule ==True:
                obs.display_state()
                print("Priority Weighted Cycle Time (PWCT): ", total_reward)
            break
    return total_reward


# Online DQN agent algorithms

def get_best_schedule(agent, env, show_schedule):
    state = env.reset()
    done = False
    total_reward  = 0

    while not done:
        action = agent.test_select_action(state)
        #print(action) # uncomment for debugging
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
    if show_schedule ==True:
        print(next_state.display_state())
        print("Priority Weighted Cycle Time(PWCT) : {}".format(total_reward))

    return total_reward

# Get the best parameters from the optimisation: Online
df_online = pd.read_csv('results_parameters_online.csv')
best_row_online = df_online.loc[df_online['total_reward'].idxmin()]

best_batch_size_online = int(best_row_online['batch_size'])
best_gamma_online = float(best_row_online['gamma'])
best_epsilon_start_online = float(best_row_online['epsilon_start'])
best_epsilon_end_online = float(best_row_online['epsilon_end'])

# Run the online DQN agent
def online_dqn(job_list, machine_list, show_schedule = False):

    env = SchedulingEnvironment(job_list, machine_list)
    input_size = 4 + (len(machine_list)*4) + len(machine_list)
    output_size = len(job_list)*len(machine_list)
    agent = DQNAgent_online(input_size, output_size, job_list, machine_list, batch_size=best_batch_size_online, gamma=best_gamma_online, epsilon_start=best_epsilon_start_online, epsilon_end=best_epsilon_end_online, epsilon_decay=0.995)

    # Training loop
    num_episodes = 6000
    loss_list = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            env.current_state = next_state
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.optimize_model()  # Get the model loss value

            total_reward = reward
            state = next_state
            if done:
                agent.update_target_network()
                break
        loss_list.append(loss)
        
        if show_schedule == True and loss != None and episode % 50 == 0 : #and episode >500
            visualize_loss(loss_list)
            print(f"Episode {episode + 1}: Model loss = {loss}")

    # Save the trained model
    agent.save_model("dqn_online.pth") 

    # Loading the trained model
    agent.load_model("dqn_online.pth")

    tot_reward = get_best_schedule(agent, env, show_schedule)
    return tot_reward


# Offline DQN agent algorithms

# Get the best parameters from the optimisation: Offline
df_offline = pd.read_csv('results_parameters_offline.csv')
best_row_offline = df_offline.loc[df_offline['total_reward'].idxmin()]

best_discount_factor_offline = float(best_row_offline['discount_factor'])
best_max_iterations_offline = int(best_row_offline['max_iterations'])
best_first_hidden_layer_offline = int(best_row_offline['first_hidden_layer'])
best_last_hidden_layer_offline = int(best_row_offline['last_hidden_layer'])

# Run the offline DQN agent
def offline_dqn(job_list, machine_list, show_schedule = False):
    offline_data = build_offline_dataset(job_list, machine_list, iterations = 100, action_agent=RandomAgent(), short_vector = False, standardise = True)

    class ZeroEstimator:
        def predict(self, X):
            return np.zeros(len(X))

    # FQI algorithm
    discount_factor = best_discount_factor_offline
    iterations = 20
    estimator = ZeroEstimator()

    state_rep = np.array(offline_data["state"].tolist())
    next_state_rep = np.array(offline_data["next_state"].tolist())

    total_rewards = []
    min_reward = 1000000

    for i in range(iterations):

        X = state_rep
        y = []
        for index, row in offline_data.iterrows():
            state, reward, next_state, is_terminal, possible_states = row
            y.append(compute_td_target(next_state, reward, is_terminal, discount_factor, estimator, possible_states))

        y = np.array(y)

        estimator = MLPRegressor(max_iter=best_max_iterations_offline, hidden_layer_sizes=(best_first_hidden_layer_offline, 128, 64, best_last_hidden_layer_offline))
        estimator.fit(X, y)

        print(f'Iteration {i+1} out of {iterations} complete')

        # compute the reward using the current estimator
        env = SchedulingEnvironment(job_list, machine_list)
        agent = DQNAgent_offline(estimator, short_vector = False)
        obs = env.reset()
        total_reward = 0
        current_state = env.reset()
        done = False
        t=0
        while not done:
            action = agent.select_action(obs)
            next_state, reward, done = env.step_id(action)
            env.current_state =next_state
            total_reward += reward
            obs = next_state
            # obs.display_state()
            if done:
                # obs.display_state()
                print("Total Reward: ", total_reward)
                total_rewards.append(total_reward)
                break

        if total_reward < min_reward:
            min_reward = total_reward
            best_estimator = copy.deepcopy(estimator)

        if show_schedule == True:
            visualize_loss(total_rewards)

    filename = 'finalized_model.sav'
    pickle.dump(best_estimator, open(filename, 'wb'))
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    env = SchedulingEnvironment(job_list, machine_list)
    agent = DQNAgent_offline(loaded_model, short_vector = False)
    obs = env.reset()
    total_reward = 0
    current_state = env.reset()
    done = False
    t=0
    while not done:
        action = agent.select_action(obs)
        next_state, reward, done = env.step_id(action)
        env.current_state =next_state
        total_reward += reward
        obs = next_state
        # obs.display_state()
        if done:
            if show_schedule == True:
                obs.display_state()
                print("Priority Weighted Cycle Time (PWCT) DQNAgent_offline: ", total_reward)
            break
    return total_reward