import numpy as np
import matplotlib.pyplot as plt

# Define the function
def func(x):
    return np.sin(2 / (10000 ** (2 * x / 512)))

def func_2(x):
    return np.cos(2 / (10000 ** (2 * x / 512)))


# Define the range of x values
x_values = np.linspace(-100, 100, 1000)

# Compute the corresponding y values
y_values = func(x_values)
y_values_2 = func_2(x_values)

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='$y = \sin\\left(\\frac{2}{10000^{\\frac{2x}{512}}}\\right)$')
plt.plot(x_values, y_values_2,label='$y = \cos\\left(\\frac{2}{10000^{\\frac{2x}{512}}}\\right)$')
plt.title('Plot of the Function $y = \sin\\left(\\frac{2}{10000^{\\frac{2x}{512}}}\\right)$')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()
