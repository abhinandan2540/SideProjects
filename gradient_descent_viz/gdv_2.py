
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# loss functions
# it is subject to change and can be of any mathematical equations
def loss(x, y):
    return x**2 + 10*y**2


# taking gradients (derivatives)
def grad(x, y):
    return np.array([2*x, 2*y])


# Gradient descent parameters
# learning rate and steps
lr = 0.15
steps = 25

# Starting point
point = np.array([4.0, 4.0])

trajectory = [point.copy()]
loss_values = [loss(point[0], point[1])]


for _ in range(steps):
    g = grad(point[0], point[1])
    point = point - lr * g
    trajectory.append(point.copy())
    loss_values.append(loss(point[0], point[1]))

trajectory = np.array(trajectory)
loss_values = np.array(loss_values)

# grid mesh creation
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = loss(X, Y)

# plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# surface
ax.plot_surface(X, Y, Z, alpha=0.6)

# gradient descent path
ax.plot(trajectory[:, 0], trajectory[:, 1], loss_values, 'r-o', linewidth=2)

# ax.set_title("3D Loss Surface with Gradient Descent Path")
ax.set_title("Surface With Gradient Descent Path")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Loss")

plt.show()
