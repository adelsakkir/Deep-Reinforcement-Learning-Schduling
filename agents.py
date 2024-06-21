import random
import copy
import numpy as np

# DQN_offline requirements
from environment import SchedulingEnvironment

# DQN_online requirements
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple

#############################################
# Definitions in agents.py: Agents classes for the scheduling environment
#
# 1. GreedyAgent1
# 2. RandomAgent
# 3. DQNAgent_offline
# 4. DQNAgent_online
#############################################


# Greedy agent

class GreedyAgent1:
    def __init__(self):
        pass

    def select_action(self, state):
        best_cost = 1000000
        greedy_action = None
        for job, machine in state.available_actions:
            if (job.priority)*(machine.end_time+machine.processing_times[job.family]) <= best_cost:
                greedy_action = (job, machine)
                best_cost = (job.priority)*(machine.end_time+machine.processing_times[job.family])
        return greedy_action


# Random agent

class RandomAgent:
    def __init__(self):
        pass

    def select_action(self, state):
        action = random.choice(state.available_actions)
        return action


# DQN offline agent

class DQNAgent_offline:
    def __init__(self, estimator, short_vector = False):
        self.estimator = estimator
        self.short_vector = short_vector

    def select_action(self, state):
        best_action = None
        best_value = float('inf')
        state_in = copy.deepcopy(state)
        # get available moves for that state and compute all the possible next states
        for action in state.available_actions_id:

            env_in = SchedulingEnvironment(state_in.remaining_jobs, state_in.machines)
            next_state, reward, done = env_in.step_id(action)

            # compute the value of the next state
            next_state_value = self.estimator.predict([next_state.get_feature_vector(short=self.short_vector)])

            # we want to select the action that minimises the value of the next state
            if next_state_value < best_value:
                best_value = next_state_value
                best_action = action

        return best_action



# DQN online agent

# Define the network class
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        # self.fc2 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, output_size)

        # self.fc1 = nn.Linear(input_size, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128, 64)
        # self.fc5 = nn.Linear(64, output_size)

    def forward(self, x):
        m = nn.LeakyReLU(0.1)
        x = m(self.fc1(x))
        x = m(self.fc2(x))
        x = m(self.fc3(x))
        x = m(self.fc4(x))
        # x = m(self.fc5(x))
        return self.fc5(x)

# Define a replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Define the DQN agent
class DQNAgent_online:
    def __init__(self, input_size, output_size, jobs, machines, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay=0.995):
    #def __init__(self, input_size, output_size, jobs, machines, batch_size=16, gamma=0.9, epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.995): # tested out previously
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(input_size, output_size).to(self.device)
        self.target_net = DQN(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayBuffer(100000)
        #self.memory = PrioritizedReplayBuffer(10000)

        self.steps_done = 0
        self.steps_done = 0
        #------------------------------------------------------------

        self.jobs = jobs
        self.machines = machines
        self.num_jobs = len(jobs)
        self.num_machines = len(machines)

    def select_action(self, state):
        """Select actions using an epsilon-greedy strategy."""

        sample = random.random()
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

        # New code: ensure that the model cannot select None as an action
        if sample > self.epsilon:
            with torch.no_grad():
                # Ensure there are available actions
                if not state.available_actions:
                    raise ValueError("No available actions to choose from.")

                # Start with a random valid action in case no action from Q-values matches
                best_action = random.choice(state.available_actions)
                best_action_value = float('inf')  # Choose a high best action value since we are minimizing

                state_format = torch.tensor(state.get_feature_vector(), dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_format)
                q_array = q_values.numpy()[0]

                for action_index in range(len(q_array)):
                    action = self.find_action_from_index(action_index)
                    # Check if the action is valid and has a better Q-value
                    if action in state.available_actions and q_array[action_index] < best_action_value:
                        best_action = action
                        best_action_value = q_array[action_index]

                action = best_action



        else:
            action = random.choice(state.available_actions)

        return action

    def find_action_from_index(self, index):
        job_id = (index%self.num_jobs) +1
        machine_id = (index//(self.num_jobs))+1
        # print(self.num_jobs, machine_id)

        return (self.jobs[job_id-1], self.machines[machine_id-1])

    def find_index_from_action(self, action):
        action_job_id = action[0].id
        action_machine_id = action[1].id

        start_index = 0
        for mach in self.machines:
            if action_machine_id == mach.id:
                break
            else:
                start_index += self.num_jobs
        index = start_index + (action_job_id-1)
        return index

    def test_select_action(self, state):
        with torch.no_grad():
            # Ensure there are available actions
            if not state.available_actions:
                raise ValueError("No available actions to choose from.")

            # Start with a random valid action in case no action from Q-values matches
            best_action = random.choice(state.available_actions)
            best_action_value = float('inf')  # Choose a high best action value since we are minimizing

            state_format = torch.tensor(state.get_feature_vector(), dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_format)
            q_array = q_values.numpy()[0]

            for action_index in range(len(q_array)):
                action = self.find_action_from_index(action_index)
                # Check if the action is valid and has a better Q-value
                if action in state.available_actions and q_array[action_index] < best_action_value:
                    best_action = action
                    best_action_value = q_array[action_index]

            action = best_action

        return action


    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state.get_feature_vector(), self.find_index_from_action(action), reward, next_state.get_feature_vector(), done)
    #------------------------------------------------------------
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # print ("Transistions: ", transitions)
        batch = Transition(*zip(*transitions))
        # print ("batch: ", batch)

        non_final_mask = torch.tensor(tuple(map(lambda s: not s, batch.done)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.tensor([s for s, d in zip(batch.next_state, batch.done) if not d],
                                             dtype=torch.float32).to(self.device)
        # print("Non final next states ", non_final_next_states)
        state_batch = torch.tensor(batch.state, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        # print("State Batch: ", state_batch)
        # print("Action Batch: ", action_batch)
        # print("Reward Batch: ", reward_batch)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # print ("Batch Size: ", self.batch_size)
        # print ("State action values: ", state_action_values)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    #------------------------------------------------------------
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
















