# Pattern recognition course | ECE AUTH | 1st project | Winter semester 2022
# Giachoudis Christos
# Kostopoulos Andreas Marios

# Importing the necessary modules
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# A priori knowledge
m1 = np.array([0.4, 0.8]) # mean values for class ω_1
m2 = np.array([1.5, 2.7]) # mean values for class ω_2
cov_mat = np.array([[1.5, 0], [0, 0.8]]) # covariance matrix for the 2 data indicators
p_1 = 0.95 # a priori probability for class ω_1
p_2 = 0.05 # a priori probability for class ω_2



# ***Task A.1***

# Initializing the random seed
random_seed = 1000

# create the distributions for x|ω1 and x|ω2 (so that you can vizualize them)
distr1 = multivariate_normal(cov = cov_mat, mean = m1, seed = random_seed)
distr2 = multivariate_normal(cov = cov_mat, mean = m2, seed = random_seed)

# Generating 500 samples out of the distributions
data1 = distr1.rvs(size = 500)
# Generating 500 samples out of the distribution
data2 = distr2.rvs(size = 500)

# Plotting the generated samples
# 2D plotting
fig = plt.figure(figsize=(10, 10), dpi=100)
fig.suptitle('Distributions of microbiological indicators', fontsize=20)
plt.plot(data1[:,0], data1[:,1], 'o', c = 'lime', markeredgewidth = 0.5, markeredgecolor = 'black', label = 'healthy')
plt.xlabel('Biological indicator a')
plt.ylabel('Biological indicator b')
plt.plot(data2[:,0], data2[:,1], 'o', c = 'firebrick', markeredgewidth = 0.5, markeredgecolor = 'black', label = 'possible cancer existence')
plt.axis('equal')
plt.legend(loc="upper left")

# Some representative values
print('Some representative values for healthy people are:')
x1 = [[-1.1, 0], [0.4, 0.8], [1.9, 1.6], [-1.1, 1.6], [1.9, 0], [1.5, 2.7]];
y1 = multivariate_normal.pdf(x1, mean = m1, cov = cov_mat);
print("[x_a = -1.1, x_b = 0]: " + str(y1[0]))
print("[x_a = 0.4, x_b = 0.8]: " + str(y1[1]))
print("[x_a = 1.9, x_b = 1.6]: " + str(y1[2]))
print("[x_a = -1.1, x_b = 1.6]: " + str(y1[3]))
print("[x_a = 1.9, x_b = 0]: " + str(y1[4]))
print("[x_a = 1,5, x_b = 2.7]: " + str(y1[5])) # to see what's happening at the none healthy field

print('Some representative values for people with possible cancer existence are:')
x2 = [[0, 1.9], [1.5, 2.7], [3, 3.5], [0, 3.5], [3, 1.9], [0.4, 0.8]];
y2 = multivariate_normal.pdf(x2, mean = m2, cov = cov_mat);
print("[x_a = 0, x_b = 1.9]: " + str(y2[0]))
print("[x_a = 1.5, x_b = 2.7]: " + str(y2[1]))
print("[x_a = 3, x_b = 3.5]: " + str(y2[2]))
print("[x_a = 0, x_b = 3.5]: " + str(y2[3]))
print("[x_a = 3, x_b = 1.9]: " + str(y2[4]))
print("[x_a = 0.4, x_b = 0.8]: " + str(y2[5])) # to see what's happening at the healthy field

print("where x_a is the first indicator and x_2 the second one.")
print("")
print("")
print("For better understanding:")

# 3D plotting
fig = plt.figure(figsize=(10, 10)) # prepare a figure
ax = fig.add_subplot(111, projection='3d') # the figure will hold a 3d plot
ax.set_title("Pdf representative values vizualization")
ax.set_xlabel("x_a indicator")
ax.set_ylabel("x_b indicator")
ax.set_zlabel("Pdf value")
for i in range(6):
  ax.scatter(x1[i][0], x1[i][1], y1[i], c = "blue")
for i in range(6):
  ax.scatter(x2[i][0], x2[i][1], y2[i], c = "red", marker = "x")
# I want to find a way to plot a line too under each point, so that it makes clearer the 3d view of the plot(!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)


# Result A.1

#We can observe that the density near the mean values is higher than the one at combinations away from them. At those distaned combinations it seems that it is highly rare to find a person with the related situation (has cancer or not). That can be easily shown at the 3d figure too. Higher blue points means healthy, higher red points means cancer. (that last one must be fixed a little!!!!!!!)

# Task A.2

# p(x) = p(x|ω1)*p(ω1) + p(x|ω2)*p(ω2)

p = y1 * p_1 + y2 * p_2
# 3D plotting
fig = plt.figure(figsize=(10, 10)) # prepare a figure
# ax = fig.add_subplot(111, projection='3d') # the figure will hold a 3d plot
ax = plt.axes(projection = '3d')
ax.set_title("Pdf representative values vizualization")
ax.set_xlabel("x_a indicator")
ax.set_ylabel("x_b indicator")
ax.set_zlabel("Pdf values for total probability distribution")
for i in range(1000):
  ax.scatter(data[i][0], data[i][1], p[i], c = "blue")

ax.view_init(35, 35)


# Task A.3

# a posteriori

# p(ω1|x) = (p(x|ω1)/p(x))*p(ω1)
# p(ω2|x) = (p(x|ω2)/p(x))*p(ω2)

p_aposteriori_1 = (y1/p) * p_1;
p_aposteriori_2 = (y2/p) * p_2;

fig = plt.figure(figsize=(10, 10)) # prepare a figure
# ax = fig.add_subplot(111, projection='3d') # the figure will hold a 3d plot
ax = plt.axes(projection = '3d')
ax.set_title("Pdf representative values vizualization")
ax.set_xlabel("x_a indicator")
ax.set_ylabel("x_b indicator")
ax.set_zlabel("Pdf values for total probability distribution")
x, y = data[:,0], data[:,1]
# for i in range(1000):
#   ax.plot_trisurf(data[i][0], data[i][1], p_aposteriori_1[i], c = "green")
# for i in range(1000):
#   ax.plot(data[i][0], data[i][1], p_aposteriori_2[i], c = "red", marker = "x")
ax.plot_trisurf(x, y, p_aposteriori_1)
ax.plot_trisurf(x, y, p_aposteriori_2)


# ax.view_init(15, 0)
