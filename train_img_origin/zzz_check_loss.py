import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, np.pi * 2, 100, endpoint=True)
y_sin = np.sin(2*x)
y_cos = np.cos(2*x)
print(len(y_cos))

plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.show()