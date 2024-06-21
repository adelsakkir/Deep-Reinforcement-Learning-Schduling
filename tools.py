import random
import numpy as np
import pandas as pd
from agents import GreedyAgent1
from environment import Job, Machine, SchedulingEnvironment
import matplotlib.pyplot as plt
from IPython import display
import copy


#############################################
# Definitions in tools.py: various functions needed to build the dataset and train the DQN agent
#
# 1. function to generate random jobs
# 2. function to generate random machines
# 3. function to build the offline dataset
# 4. function to compute the TD target for the offline DQN agent
# 5. function to visualize the loss during training of the online DQN agent
#############################################


def generate_random_jobs(num_jobs, num_families):
    families = ['F' + str(i) for i in range(1, num_families + 1)]
    job_list = []
    for i in range(1, num_jobs + 1):
        # job = Job(id=i, priority=random.randint(1, 10), family=random.choice(families))
        job = Job(id=i, priority=random.choice([1, 0.1, 0.5]), family=random.choice(families))
        job_list.append(job)
    return job_list

def generate_random_machines(num_machines, num_families):
    families = ['F' + str(i) for i in range(1, num_families + 1)]
    machine_list = []

    for i in range(1, num_machines + 1):
        num_families_for_machine = random.randint(1, num_families)
        families_for_machine = random.sample(families, num_families_for_machine)
        # family_dict = {family: random.randint(1, 10) for family in families_for_machine} #10-50
        family_dict = {family: random.choice([30, 60 , 90, 120]) for family in families_for_machine}
        machine = Machine(id=i, family_dict=family_dict)
        machine_list.append(machine)

    return machine_list

def build_offline_dataset(job_list, machine_list, iterations = 100, short_vector = False, action_agent = GreedyAgent1(), standardise = True):

    env = SchedulingEnvironment(job_list, machine_list)
    offline_data = []

    for i in range(iterations):
        done = False
        env.reset()
        while not done:

            current_state = env.current_state
            state_representation = current_state.get_feature_vector(short=short_vector, standardise = standardise)
            # take an action based on the agent's policy with 0.9 probability, and a random action with 0.1 probability
            if random.random() < 0.9:
                action = action_agent.select_action(env.current_state)
            else:
                action = random.choice(env.current_state.available_actions)

            next_state, reward, done = env.step(action)
            next_state_representation = next_state.get_feature_vector(short=short_vector, standardise = standardise)

            # # Update the current state
            env.current_state = next_state

            # # get all possible future states
            possible_next_states = []

            # make a copy of the state
            current_state_copy = copy.deepcopy(next_state)
            job_list_copy = copy.deepcopy(env.jobs)
            machine_list_copy = copy.deepcopy(env.machines)

            env_copy = SchedulingEnvironment(job_list_copy, machine_list_copy)
            env_copy.current_state = current_state_copy

            for action_id in env.current_state.available_actions_id:

                env_copy = SchedulingEnvironment(job_list_copy, machine_list_copy)
                env_copy.current_state = current_state_copy

                future_state, reward, done = env_copy.step_id(action_id)

                # get a feature vector representation of the state
                future_state_representation = future_state.get_feature_vector(short=short_vector, standardise = standardise)
                possible_next_states.append(future_state_representation)


            # add the observation to the dataset
            offline_data.append([state_representation, reward, next_state_representation, done, possible_next_states])


    return pd.DataFrame(offline_data, columns=["state", "reward", "next_state", "is_terminal", "possible_next_states"])


def compute_td_target(next_state, reward, is_terminal, discount_factor, estimator, possible_states):
    if is_terminal:
        return reward
    else:
        # compute the value of the next state
        min_value = 1000000000
        for possible_state in possible_states:

            possible_value = estimator.predict([possible_state])[0]
            if possible_value < min_value:
                min_value = possible_value
        return reward + discount_factor * min_value

def visualize_loss(loss_values):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Loss Value')
    plt.plot(range(len(loss_values)), loss_values)
    plt.ylim(ymin=0)
    plt.show(block=False)
