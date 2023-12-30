import numpy as np
import matplotlib.pyplot as plt

def sine_map(x, lmbda):
    return lmbda * np.sin(np.pi * x)

lmbda_values = np.linspace(0.6, 1, 1000)
x_values = [[] for _ in range(len(lmbda_values))]

initial_condition = 0.1

for i, lmbda in enumerate(lmbda_values):
    x = initial_condition
    for _ in range(1000):
        x = sine_map(x, lmbda)
    for _ in range(100):
        x = sine_map(x, lmbda)
        x_values[i].append(x)

plt.figure(figsize=(8, 6))
for i, lmbda in enumerate(lmbda_values):
    plt.plot([lmbda] * len(x_values[i]), x_values[i], ',k')
plt.title('Bifurcation diagram')
plt.xlabel('λ')
plt.ylabel('x_n')
plt.show()