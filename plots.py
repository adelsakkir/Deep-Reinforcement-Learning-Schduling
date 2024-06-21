import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


#############################################
# Definitions in plots.py: Visualisations of the state features
#
# 1. Visualisation of the state feature S_J (jobs per family)
# 2. Visualisation of the state feature S_B (machine busyness matrix)
# 3. Visualisation of the state feature S_P (priority weighted processing time vector)
#############################################


# 1. Jobs per family

data = [
    np.array([[2, 3, 1, 4]]),
    np.array([[2, 3, 0, 4]]),
    np.array([[2, 3, 0, 3]])
]

titles = ['State 1', 'State 2', 'State 3']
colors = [['#9ecae1', '#f7fbff', '#9ecae1', '#f7fbff'],  
          ['#9ecae1', '#f7fbff', '#ef3b2c', '#f7fbff'],  
          ['#9ecae1', '#f7fbff', '#9ecae1', '#ef3b2c']] 

fig, axes = plt.subplots(1, 3, figsize=(10, 3))  # Adjust figsize to suit your needs

for idx, ax in enumerate(axes):
    ax.set_title(titles[idx])
    matrix = data[idx]
    num_cols = matrix.shape[1]
    
    # Create a colored rectangle for each column
    for j in range(num_cols):
        color = colors[idx][j]
        rect = patches.Rectangle((j, 0), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        # Adding text
        ax.text(j + 0.5, 0.5, str(matrix[0, j]), color='black', ha='center', va='center')
    
    # Set limits and aspect
    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0.5, num_cols, 1))
    ax.set_yticks([0.5])
    ax.set_xticklabels(['F_1', 'F_2', 'F_3', 'F_4']) 
    ax.set_yticklabels(['']) 

plt.tight_layout()

# Save the figure
plt.savefig('plots/jobs_per_family.png')
plt.close()


# 2. Machine busyness matrix

data = [
    np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]),
    np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0]
    ]),
    np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0]
    ])
]


titles = ['State 1', 'State 2', 'State 3']

# Define colors for each cell in the matrix, needs to be adjusted to 5x4
colors = [
    [['#9ecae1', '#f7fbff', '#9ecae1', '#f7fbff']] * 5, 
    [['#9ecae1', '#f7fbff', '#9ecae1', '#f7fbff'], ['#9ecae1', '#f7fbff', '#9ecae1', '#f7fbff'], ['#9ecae1', '#f7fbff', '#9ecae1', '#f7fbff'], ['#9ecae1', '#f7fbff', '#ef3b2c', '#f7fbff'], ['#9ecae1', '#f7fbff', '#9ecae1', '#f7fbff']],
    [['#9ecae1', '#f7fbff', '#9ecae1', '#ef3b2c'], ['#9ecae1', '#f7fbff', '#9ecae1', '#f7fbff'], ['#9ecae1', '#f7fbff', '#9ecae1', '#f7fbff'], ['#9ecae1', '#f7fbff', '#9ecae1', '#f7fbff'], ['#9ecae1', '#f7fbff', '#9ecae1', '#f7fbff']],
]

fig, axes = plt.subplots(1, 3, figsize=(10, 6)) 

for idx, ax in enumerate(axes):
    ax.set_title(titles[idx])
    matrix = data[idx]
    num_rows, num_cols = matrix.shape
    
    # Create a colored rectangle for each cell
    for i in range(num_rows):
        for j in range(num_cols):
            color = colors[idx][i][j]
            rect = patches.Rectangle((j, num_rows-i-1), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            ax.text(j + 0.5, num_rows - i - 0.5, str(matrix[i, j]), color='black', ha='center', va='center')
    
    # Set limits and aspect
    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0.5, num_cols, 1))
    ax.set_yticks(np.arange(0.5, num_rows, 1))
    ax.set_xticklabels(['F_1', 'F_2', 'F_3', 'F_4']) 
    ax.set_yticklabels(['M_1', 'M_2', 'M_3', 'M_4', 'M_5'])

plt.tight_layout()

# Save the figure
plt.savefig('plots/machine_busyness.png')
plt.close()



# 3. Priority weighted processing time vector

data = [
    np.array([[0, 0, 0, 0, 0]]),
    np.array([[0, 15, 0, 0, 0]]),
    np.array([[0, 0, 0, 0, 60]])
]

titles = ['State 1', 'State 2', 'State 3']
colors = [['#9ecae1', '#f7fbff', '#9ecae1', '#f7fbff', '#9ecae1'],  
          ['#9ecae1', '#ef3b2c', '#9ecae1', '#f7fbff', '#9ecae1'],  
          ['#9ecae1', '#f7fbff', '#9ecae1', '#f7fbff', '#ef3b2c']] 

fig, axes = plt.subplots(1, 3, figsize=(10, 3)) 

for idx, ax in enumerate(axes):
    ax.set_title(titles[idx])
    matrix = data[idx]
    num_cols = matrix.shape[1]
    
    # Create a colored rectangle for each column
    for j in range(num_cols):
        color = colors[idx][j]
        rect = patches.Rectangle((j, 0), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(j + 0.5, 0.5, str(matrix[0, j]), color='black', ha='center', va='center')
    
    # Set limits and aspect
    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0.5, num_cols, 1))
    ax.set_yticks([0.5])
    ax.set_xticklabels(['M_1', 'M_2', 'M_3', 'M_4', 'M_5']) 
    ax.set_yticklabels([''])  

plt.tight_layout()

# Save the figure
plt.savefig('plots/priority_processing_time.png')
plt.close()