import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

def read_data(file_path):
    categories = []
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.split()
            categories.append(values[0])
            data.append(list(map(float, values[1:])))
    return categories, np.array(data)

# Read data from file
categories, data = read_data('exp-fig-11.txt')

# Data for each series
cinm_4d = data[:, 0]
cinm_opt_4d = data[:, 1]
cinm_8d = data[:, 2]
cinm_opt_8d = data[:, 3]
cinm_16d = data[:, 4]
cinm_opt_16d = data[:, 5]

# X-axis coordinates
x = np.arange(len(categories))

# Bar width
width = 0.15

# Colormap
colors = cm.GnBu(np.linspace(0.3, 1, 6))

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Bars with custom colors
rects1 = ax.bar(x - 2.5*width, cinm_4d, width, label='cinm-4d', color=colors[0])
rects2 = ax.bar(x - 1.5*width, cinm_opt_4d, width, label='cinm-opt-4d', color=colors[1])
rects3 = ax.bar(x - 0.5*width, cinm_8d, width, label='cinm-8d', color=colors[2])
rects4 = ax.bar(x + 0.5*width, cinm_opt_8d, width, label='cinm-opt-8d', color=colors[3])
rects5 = ax.bar(x + 1.5*width, cinm_16d, width, label='cinm-16d', color=colors[4])
rects6 = ax.bar(x + 2.5*width, cinm_opt_16d, width, label='cinm-opt-16d', color=colors[5])

# Labels and formatting
ax.set_xlabel('')
ax.set_ylabel('Execution time (ms) (log scale)')
ax.set_title('Impact of optimizations on performance')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.set_yscale('log')
ax.legend()

ax.grid(True, axis='y', which='major', ls='--', linewidth=0.5)
ax.grid(False, axis='x')

# Layout adjustments
plt.tight_layout()

plt.savefig('fig11.pdf')
plt.show()
