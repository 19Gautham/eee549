import math
import numpy as np
import matplotlib.pyplot as plt

def get_radmacher_cdf_val(x):
    res = 0
    if x < -1:
        res = 0
    elif -1 <= x < 1:
        res = 0.5
    else:
        res = 1
    return res

# range of Y^(k) values the CDF will be plotted against
x_vals = np.arange(-5, 5, 0.01)
# arbitrary  number of samples to be generated
n = 1000

for k in [1, 8, 64, 512]:
    radmacher_array = np.random.choice(a=[-1,1], size=(n, k))
    # print(radmacher_array.shape)
    y_k_values = radmacher_array.sum(axis=1) * (1/math.sqrt(k))
    # print(y_k_values)
    cdf_values = []
    for val in x_vals:
        cdf_values.append(sum(y_k_values <= val)/n)
    plt.plot(x_vals, cdf_values, label=f'k = {k}')

# Radmacher CDF
cdf_values = []
for val in x_vals:
    cdf_values.append(get_radmacher_cdf_val(val))
plt.plot(x_vals, cdf_values, label="Radmacher", linestyle='--', color='black')

# Gaussian distribution
cdf_values = []
gaussian_samples = sorted(np.random.standard_normal(size=3000))
for val in x_vals:
    cdf_values.append(sum(gaussian_samples <= val) / 3000)
plt.plot(x_vals, cdf_values, label="Gaussian", color='purple')

plt.xticks(np.arange(-5, 6, 0.5))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title('Empirical CDF plots')
plt.xlabel('Observations')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()