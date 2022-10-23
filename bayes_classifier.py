# Pattern recognition course | ECE AUTH | 1st project
# Giachoudis Christos | 9912 | Winter semester 2022

# Importing the necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math

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

# Generating 5000 samples out of the distributions
data1 = distr1.rvs(size = 500)
# Generating 5000 samples out of the distribution
data2 = distr2.rvs(size = 500)

# Plotting the generated samples
# 2D plotting
fig = plt.figure(figsize=(10, 10), dpi=100)
fig.suptitle('Distributions of microbiological indicators', fontsize=20)
plt.plot(data1[:,0], data1[:,1], 'o', c = 'lime', markeredgewidth = 0.5, markeredgecolor = 'black', label = 'healthy')
plt.xlabel('x_a')
plt.ylabel('x_b')
plt.plot(data2[:,0], data2[:,1], 'o', c = 'firebrick', markeredgewidth = 0.5, markeredgecolor = 'black', label = 'possible cancer existence')
plt.axis('equal')
plt.legend(loc="upper left")

# Some representative values


print('Some representative values for healthy people are:')

print('Some representative values for people with possible cancer existence are:')
