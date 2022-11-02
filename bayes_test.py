# print('Test')
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from scipy.stats import norm
import statistics

# # A priori knowledge
m1 = np.array([0.4, 0.8]) # mean values for class ω_1
m2 = np.array([1.5, 2.7]) # mean values for class ω_2
cov_mat = np.array([[1.5, 0], [0, 0.8]]) # covariance matrix for the 2 data indicators
p_1 = 0.95 # a priori probability for class ω_1
p_2 = 0.05 # a priori probability for class ω_2

# # ***Task A.1***

def linspace(start, stop, step=1.):
  """
    Like np.linspace but uses step instead of num
    This is inclusive to stop, so if start=1, stop=3, step=0.5
    Output is: array([1., 1.5, 2., 2.5, 3.])
  """
  return np.linspace(start, stop, int((stop - start) / step + 1))

# # Initializing the random seed
random_seed = 1000


#Data
distr1 = multivariate_normal(cov = cov_mat, mean = m1, seed = random_seed)
mean1, mean2 = m1[0], m1[1]
sigma1, sigma2 = cov_mat[0, 0], cov_mat[1, 1]

x1 = linspace(mean1 - 3*sigma1, mean1 + 3*sigma1, step=0.1)
x2 = linspace(mean2 - 3*sigma2, mean2 + 3*sigma2, step=0.1)
X1, X2 = np.meshgrid(x1, x2)

pdf1 = np.zeros(X1.shape)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        pdf1[i, j] = distr1.pdf([X1[i, j], X2[i, j]])

distr2 = multivariate_normal(cov = cov_mat, mean = m2, seed = random_seed)
mean3, mean4 = m2[0], m2[1]

y1 = linspace(mean3 - 3*sigma1, mean3 + 3*sigma1, step=0.1)
y2 = linspace(mean4 - 3*sigma2, mean4 + 3*sigma2, step=0.1)
Y1, Y2 = np.meshgrid(y1, y2)

pdf2 = np.zeros(Y1.shape)
for i in range(Y1.shape[0]):
    for j in range(Y1.shape[1]):
        pdf2[i, j] = distr2.pdf([Y1[i, j], Y2[i, j]])

#plotting
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection = '3d')
ax.plot_surface(X1, X2, pdf1)
ax.plot_surface(Y1, Y2, pdf2)
plt.show()




