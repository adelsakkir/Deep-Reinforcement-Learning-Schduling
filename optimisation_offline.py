# Optimise the parameters of the offline model

from environment import SchedulingEnvironment
from tqdm import tqdm
import pandas as pd
from tools import generate_random_jobs, generate_random_machines, build_offline_dataset, compute_td_target, visualize_loss
import copy
import random
import numpy as np
from sklearn.neural_network import MLPRegressor
import pickle
from agents import RandomAgent
from agents import DQNAgent_offline

# Generate random jobs and machines
random.seed(123)
job_list = generate_random_jobs(10, 4)
machine_list = generate_random_machines(5, 4)


discount_factor_values = [0.9, 0.95, 0.99]
max_iterations_values = [500, 1000, 2000]
first_hidden_layer_values = [128, 265, 512]
last_hidden_layer_values = [8, 16, 32]


num_combinations = len(discount_factor_values) * len(max_iterations_values) * len(first_hidden_layer_values) * len(last_hidden_layer_values)

current_iteration = 0

columns = ["discount_factor", "max_iterations", "first_hidden_layer", "last_hidden_layer", "total_reward"]
results_parameters_offline = pd.DataFrame(columns=columns)

offline_data = build_offline_dataset(job_list, machine_list, iterations = 100, short_vector = False, action_agent = RandomAgent(), standardise = True)

class ZeroEstimator:
    def predict(self, X):
        return np.zeros(len(X))

# Find the best parameters for the offline model
for discount_factor in discount_factor_values:
    for max_iterations in max_iterations_values:
        for first_hidden_layer in first_hidden_layer_values:
            for last_hidden_layer in last_hidden_layer_values:
                current_iteration += 1

                print(f"Iteration {current_iteration}/{num_combinations}; Testing combination {discount_factor}, {max_iterations}, {first_hidden_layer}, {last_hidden_layer}")

                # FQI algorithm
                iterations = 20
                estimator = ZeroEstimator()

                state_rep = np.array(offline_data["state"].tolist())
                next_state_rep = np.array(offline_data["next_state"].tolist())

                total_rewards = []
                min_reward = 1000000

                for i in tqdm(range(iterations), desc="Training Progress"):

                    X = state_rep
                    y = []
                    for index, row in offline_data.iterrows():
                        state, reward, next_state, is_terminal, possible_states = row
                        y.append(compute_td_target(next_state, reward, is_terminal, discount_factor, estimator, possible_states))
                    
                    y = np.array(y)

                    estimator = MLPRegressor(max_iter=max_iterations, hidden_layer_sizes=(first_hidden_layer, 128, 64, last_hidden_layer), early_stopping = True)
                    estimator.fit(X, y)

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
                        if done:
                            #print("Total Reward: ", total_reward)
                            total_rewards.append(total_reward)
                            break
                    
                    if total_reward < min_reward:
                        min_reward = total_reward
                        best_estimator = copy.deepcopy(estimator)
                
                # save the estimator to a file
                filename = 'temporary_model.sav'
                pickle.dump(best_estimator, open(filename, 'wb'))

                # load the model from the file
                filename = 'temporary_model.sav'
                loaded_model = pickle.load(open(filename, 'rb'))

                # Test the model
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
                    if done:
                        break

                # Add the results to the dataframe
                results_parameters_offline = results_parameters_offline.append({"discount_factor": discount_factor, "max_iterations": max_iterations, "first_hidden_layer": first_hidden_layer, "last_hidden_layer": last_hidden_layer, "total_reward": total_reward}, ignore_index=True)
                print("Total Reward DQNAgent_offline: {}".format(total_reward))

# Save the results to a csv file
results_parameters_offline.to_csv("results_parameters_offline.csv")
print("Results saved to results_parameters_offline.csv")