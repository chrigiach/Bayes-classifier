import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.integrate import trapz
import statistics
from matplotlib import cm

# # A priori knowledge
m1 = np.array([0.4, 0.8]) # mean values for class ω_1
m2 = np.array([1.5, 2.7]) # mean values for class ω_2
cov_mat1 = np.array([[1.5, 0], [0, 0.8]]) # covariance matrix for the first class ω1
cov_mat2 = cov_mat1/4
# print(cov_mat1, end="\n\n")
# print(cov_mat2)
p_1 = 0.95 # a priori probability for class ω_1
p_2 = 0.05 # a priori probability for class ω_2

#Function that configures the typical numpy.linespace function so that we can define the step,
#instead of the numbers of samples (num)
def linspace(start, stop, step=1.):
  """
    Like np.linspace but uses step instead of num
    This is inclusive to stop, so if start=1, stop=3, step=0.5
    Output is: array([1., 1.5, 2., 2.5, 3.])
  """
  return np.linspace(start, stop, int((stop - start) / step + 1))

# # ***Task B.1***
# # Initializing the random seed
random_seed = 1000

#Gaussian distribution for ω1 class "healthy people"
distr1 = multivariate_normal(cov = cov_mat1, mean = m1, seed = random_seed)
#Gaussian distribution for ω2 class "possible existence of cancer"
distr2 = multivariate_normal(cov = cov_mat2, mean = m2, seed = random_seed)

#Setting as sigma_i the main diagonal values of the covariance matrix of the first class ω1
sigma1, sigma2 = cov_mat1[0, 0], cov_mat1[1, 1]

# We make the values of the x vector, where x = (x1, x2).
# We take x1 in a range where: 
#   minimum value = min_value(of the μ1 and μ2 first value) - 3*sigma1
#   maximum value = max_value(of the μ1 and μ2 first value) + 3*sigma1
# AS for the x2 on the other hand we take value in a range where:
#   minimum value = min_value(of the μ1 and μ2 second value) - 3*sigma2
#   maximum value = max_value(of the μ1 and μ2 second value) + 3*sigma2
# We followed this procedure based on the method as described on rederence[1]
dataset_step = 0.1
x1 = linspace(m1[0] - 3*sigma1, m2[0] + 3*sigma1, step=dataset_step)
x2 = linspace(m1[1] - 3*sigma1, m2[1] + 3*sigma1, step=dataset_step)
X1, X2 = np.meshgrid(x1, x2)

#We form the pdf for the Gaussian distribution of the first class ω1
pdf1 = np.zeros(X1.shape)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        pdf1[i, j] = distr1.pdf([X1[i, j], X2[i, j]])

#We form the pdf for the Gaussian distribution of the second class ω2
pdf2 = np.zeros(X1.shape)
for i in range(X1.shape[0]):
  for j in range(X1.shape[1]):
    pdf2[i, j] = distr2.pdf([X1[i, j], X2[i, j]])


#plotting in the same 3-D figure the two Probability Density Functions
fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection = '3d')
plt.title("p(x|ω1) and p(x|ω2) for descrete x values following a Gaussian Distribution with μ1=(0.4, 0.8), μ2=(1.5, 2.7) and Σ1=([1.5, 0], [0, 0.8]), Σ2=([0.375, 0], [0, 0.2])")
plt.xlabel("x_a biological indicator")
plt.ylabel("x_b biological indicator")
ax.set_zlabel("pdf value")
c1 = ax.plot_surface(X1, X2, pdf1, color="green", label="PDF of class ω1")
c1._facecolors2d=c1._facecolor3d
c1._edgecolors2d=c1._edgecolor3d
c2 = ax.plot_surface(X1, X2, pdf2, color="red", label="PDF of class ω2")
c2._facecolors2d = c2._facecolor3d
c2._edgecolors2d = c2._edgecolor3d
plt.legend(loc="upper right")
plt.show()

# # ***Task B.2***
# Total Probability: P(x) = p(x|ω1)*P(ω1) + p(x|ω2)*P(ω2)
p = np.array(pdf1 * p_1 + pdf2 * p_2)

#Plotting the total PDF in a 3-D figure
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection = '3d')
plt.title("Total Probability Distribution")
plt.xlabel("x_a biological indicator")
plt.ylabel("x_b biological indicator")
ax.set_zlabel("Value of total pdf")
c1 = ax.plot_surface(X1, X2, p, label="PDF of Total Distribution")
c1._facecolors2d=c1._facecolor3d
c1._edgecolors2d=c1._edgecolor3d
plt.legend(loc="upper right")
plt.show()