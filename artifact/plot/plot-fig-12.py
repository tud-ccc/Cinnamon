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
categories, data = read_data('exp-fig-12.txt')


# Data for each series
# cpu_opt = data[:, 0]
prim_4d = data[:, 0]
cinm_opt_4d = data[:, 1]
prim_8d = data[:, 2]
cinm_opt_8d = data[:, 3]
prim_16d = data[:, 4]
cinm_opt_16d = data[:, 5]

# X-axis coordinates
x = np.arange(len(categories))

# Bar width
width = 0.12

# Colormap
colors = cm.GnBu(np.linspace(0.3, 1, 7))

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Bars with custom colors
# rects1 = ax.bar(x - 3*width, cpu_opt, width, label='cpu-opt', color=colors[0])
rects2 = ax.bar(x - 2*width, prim_4d, width, label='prim-4d', color=colors[1])
rects3 = ax.bar(x - width, cinm_opt_4d, width, label='cinm-opt-4d', color=colors[2])
rects4 = ax.bar(x, prim_8d, width, label='prim-8d', color=colors[3])
rects5 = ax.bar(x + width, cinm_opt_8d, width, label='cinm-opt-8d', color=colors[4])
rects6 = ax.bar(x + 2*width, prim_16d, width, label='prim-16d', color=colors[5])
rects7 = ax.bar(x + 3*width, cinm_opt_16d, width, label='cinm-opt-16d', color=colors[6])

# Labels and formatting
ax.set_xlabel('')
ax.set_ylabel('Execution time (ms) (log scale)')
ax.set_title('Performance comparison cinm-opt-nd and prime-nd')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.set_yscale('log')
ax.legend()

ax.grid(True, axis='y', which='major', ls='--', linewidth=0.5)
ax.grid(False, axis='x')

# Layout adjustments
plt.tight_layout()

plt.savefig('fig12.pdf')
plt.show()
