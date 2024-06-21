# Optimise the parameters of the online model

from environment import SchedulingEnvironment
from agents import DQNAgent_online
from tqdm import tqdm
import pandas as pd
import random
from tools import generate_random_jobs, generate_random_machines


# Generate random jobs and machines
random.seed(123)
job_list = generate_random_jobs(10, 4)
machine_list = generate_random_machines(5, 4)


# 81 combinations
batch_size_values = [16, 32, 64]
gamma_values = [0.9, 0.95, 0.99]
epsilon_start_values = [1, 0.9, 0.5]
epsilon_end_values = [0.1, 0.01, 0.001]

num_combinations = len(batch_size_values) * len(gamma_values) * len(epsilon_start_values) * len(epsilon_end_values)

current_iteration = 0

columns = ["batch_size", "gamma", "epsilon_start", "epsilon_end", "total_reward"]
results_parameters_online = pd.DataFrame(columns=columns)

# Find the best parameters for the online model
for batch_size in batch_size_values:
    for gamma in gamma_values:
        for epsilon_start in epsilon_start_values:
            for epsilon_end in epsilon_end_values:
                current_iteration += 1

                print(f"Iteration {current_iteration}/{num_combinations}; Testing combination {batch_size}, {gamma}, {epsilon_start}, {epsilon_end}")
                env = SchedulingEnvironment(job_list, machine_list)
                input_size = 4 + (len(machine_list)*4) + len(machine_list)
                output_size = len(job_list)*len(machine_list)
                agent = DQNAgent_online(input_size, output_size, job_list, machine_list, batch_size=batch_size, gamma=gamma, epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_decay=0.995)

                # Training loop
                num_episodes = 6000
                loss_list = []
                for episode in tqdm(range(num_episodes), desc="Training Progress"):
                    state = env.reset()
                    total_reward = 0

                    while True:
                        action = agent.select_action(state)
                        next_state, reward, done = env.step(action)

                        env.current_state = next_state
                        agent.store_transition(state, action, reward, next_state, done)
                        loss = agent.optimize_model()

                        total_reward = reward
                        state = next_state
                        if done:
                            agent.update_target_network()
                            break
                    loss_list.append(loss)

                # Test the model
                def get_best_schedule(agent, env):
                    state = env.reset()
                    done = False
                    total_reward  = 0

                    while not done:
                        action = agent.test_select_action(state)
                        next_state, reward, done = env.step(action)
                        total_reward += reward
                        state = next_state
                    #print(next_state.display_state())

                    return total_reward
                
                tot_reward = get_best_schedule(agent, env)
                
                # Add the results to the dataframe
                results_parameters_online = results_parameters_online.append({"batch_size": batch_size, "gamma": gamma, "epsilon_start": epsilon_start, "epsilon_end": epsilon_end, "total_reward": tot_reward}, ignore_index=True)
                print("Total Reward DQNAgent_online: {}".format(tot_reward))
                  

# Save the results to a CSV file
results_parameters_online.to_csv("results_parameters_online.csv")
print("Results saved to results_parameters_online.csv")
