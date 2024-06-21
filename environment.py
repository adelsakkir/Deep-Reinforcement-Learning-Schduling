import random
import copy
import numpy as np

#############################################
# Definitions in environment.py: job, machine, state, and environment classes for the scheduling problem
#
# 1. Job class
# 2. Machine class
# 3. State class
# 4. SchedulingEnvironment class
#############################################

def standardise_vector(vector):

    if np.max(vector) != 0:
        vector_stand = vector / np.max(vector)
    else:
        vector_stand = vector

    return vector_stand

class Job:
    def __init__(self, id, priority, family):
        self.id = id
        self.priority = priority
        self.family = family
        self.allocation_status = 0
        self.machine_allocated = 0
        self.start_time = None
        self.end_time = None

# Define the Machine class
class Machine:
    def __init__(self, id, family_dict):
        self.id = id
        self.families = list(family_dict.keys())
        self.processing_times = family_dict
        self.schedule = []
        self.end_time = 0
        self.busyness = {family: 0 for family in self.families}

# Define the State class
class State:
    def __init__(self, jobs, machines):
        self.machines = machines
        self.remaining_jobs =  jobs

        self.available_actions = self.get_available_actions()
        self.available_actions_id = self.get_available_actions_id()
        self.all_families = self.get_all_families()

        self.jobs_per_family = self.create_jobs_per_family_vector()
        self.machine_busyness_matrix = self.create_machine_busyness_matrix()
        self.priority_processing_time = self.create_priority_processing_time_vector()
        self.priority_matrix = self.create_priority_matrix()

    def get_available_actions(self):
        available_actions = []
        for job in self.remaining_jobs:
            for machine in self.machines:
                if job.family in machine.families:
                    available_actions.append((job, machine))
        return available_actions

    # get ids of available actions
    def get_available_actions_id(self):
        available_actions = []
        for job in self.remaining_jobs:
            for machine in self.machines:
                if job.family in machine.families:
                    # get ids of job and machine
                    job_id = job.id
                    machine_id = machine.id
                    available_actions.append((job_id, machine_id))
        return available_actions

    def print_available_actions(self):
        for job, machine in self.available_actions:
            print(f"Job {job.id}   Machine {machine.id}")

    def get_all_families(self):
        all_families = []
        for machine in self.machines:
            all_families += machine.families
        return sorted(list(set(all_families)))

    def count_jobs_per_family(self):
        job_counts_dict = {family: 0 for family in self.all_families}

        for job in self.remaining_jobs:
            family = job.family

            if family not in job_counts_dict:
                job_counts_dict[family] = 0

            job_counts_dict[family] += 1

        return job_counts_dict

    def create_jobs_per_family_vector(self, standardise=False):
        job_counts_dict = self.count_jobs_per_family()
        job_counts_list = [job_counts_dict[family] for family in job_counts_dict]

        vector = np.array(job_counts_list)
        if standardise:
            vector = standardise_vector(vector)

        return vector

    def create_machine_busyness_matrix(self, standardise=False):
        num_families = len(self.all_families)
        num_machines = len(self.machines)

        machine_busyness_matrix = np.zeros((num_machines, num_families))

        for machine_index, machine in enumerate(self.machines):
            for family_index, family in enumerate(self.all_families):
                if family in machine.busyness:
                    machine_busyness_matrix[machine_index][family_index] = machine.busyness[family]

        machine_busyness_matrix = machine_busyness_matrix.flatten()

        if standardise:
            machine_busyness_matrix = standardise_vector(machine_busyness_matrix)

        return machine_busyness_matrix

    def create_priority_matrix(self, standardise=False):
        # this is a matrix of size (num_machines, num_families) that contains the reward of jobs from family on each machine under current allocation
        num_families = len(self.all_families)
        num_machines = len(self.machines)

        priority_matrix = np.zeros((num_machines, num_families))

        # Populate the matrix using the busyness dictionary from each machine
        for machine_index, machine in enumerate(self.machines):
            for family_index, family in enumerate(self.all_families):
                for job in machine.schedule:
                    if job.family == family:
                            priority_matrix[machine_index][family_index] += job.end_time * job.priority


        # flatten and standardise the matrix
        vector = priority_matrix.flatten()
        if standardise:
            vector = standardise_vector(vector)

        return vector

    def create_priority_processing_time_vector(self, standardise=False):
        num_machines = len(self.machines)
        priority_processing_time = np.zeros(num_machines)

        for i, machine in enumerate(self.machines):
            for job in machine.schedule:
                if job != 0:
                    family = job.family
                    priority_processing_time[i] += machine.processing_times[family] * job.priority
                    priority_processing_time[i] += job.end_time * job.priority

        vector = np.array(priority_processing_time)
        if standardise:
            vector = standardise_vector(vector)

        return vector

    def get_feature_vector(self, short=False, standardise=False):
        if short:
            return self.create_priority_matrix(standardise)
        if standardise:
            return np.concatenate((self.create_jobs_per_family_vector(standardise), self.create_machine_busyness_matrix(standardise),
                                   self.create_priority_processing_time_vector(standardise)))
        else:
            return np.concatenate((self.jobs_per_family, self.machine_busyness_matrix, self.priority_processing_time))

    def display_state(self):
        print('Remaining jobs:', [job.id for job in self.remaining_jobs])

        print('Machines:')
        for machine in self.machines:
            schedule_ids = []
            for job in machine.schedule:
                if job == 0:
                    schedule_ids.append(0)
                else:
                    schedule_ids.append(job.id)

            print(f'Machine {machine.id}: {schedule_ids}')

        print('Jobs per family:', self.jobs_per_family)
        print('Machine busyness:', self.machine_busyness_matrix)
        print('Priority processing time:', self.priority_processing_time)

# Define the SchedulingEnvironment class
class SchedulingEnvironment:
    def __init__(self, jobs, machines):
        self.machines = machines
        self.jobs = jobs
        self.current_state = State(self.jobs, self.machines)

    def reset(self):
        for machine in self.machines:
            machine.schedule = []
            machine.end_time = 0
            machine.busyness = {family: 0 for family in machine.families}

        for each_job in self.jobs:
            each_job.allocation_status = 0
            each_job.machine_allocated = 0
            each_job.start_time = None
            each_job.end_time = None

        self.current_state = State(self.jobs, self.machines)
        return self.current_state

    def step(self, action):
        """
        Parameters:  action (a tuple of (job, machine)))
        Returns: next_state, reward, done
        """
        done = False

        job, machine = action

        machine.schedule.append(job)

        job.allocation_status = 1
        job.machine_allocated = machine

        job.start_time = machine.end_time
        job.end_time = job.start_time + machine.processing_times[job.family]
        machine.end_time += machine.processing_times[job.family]

        if job.family in machine.busyness:
            machine.busyness[job.family] += 1
        else:
            machine.busyness[job.family] = 1

        reward = 0

        updated_remaining_jobs = [j for j in self.current_state.remaining_jobs if j.allocation_status == 0]
        updated_machines = list(self.machines)

        next_state = State(updated_remaining_jobs, updated_machines)

        if len(next_state.remaining_jobs)==0:
            done = True

        reward = (job.priority) * job.end_time
        #reward = job.end_time
        self.current_state = next_state

        return next_state, reward, done

    def step_id(self, action_ids):
        """
        Parameters:  action (a tuple of (job_id, machine_id)))
        Returns: next_state, reward, done
        """
        done = False
        job_id, machine_id = action_ids
        job = [job for job in self.jobs if job.id == job_id][0]
        machine = [machine for machine in self.machines if machine.id == machine_id][0]

        return self.step((job, machine))
