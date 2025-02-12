import numpy as np
import matplotlib.pyplot as plt

def get_g_x_value(x):
    return 2 * np.sin(3 + 10 * np.pi * x) * np.exp((x - 1.5) ** 3)

def get_mean(vals):
    return np.mean(vals)

def get_square_diff(x, y):
    return (x-y) ** 2
def err_plotter(n, sigma_square, m_vals, x):

    # np.random.seed(0) # for reproducability, but I have excluded this for now
    y = get_g_x_value(x) + np.random.normal(scale=np.sqrt(sigma_square), loc=0, size=n)

    empirical_err = list()
    bias_square_err = list()
    variance_err = list()
    total_err = list()

    for m in m_vals:

        steps = n // m
        bias_square = variance = empirical_error = 0

        for j in range(1, steps + 1):
            c_j = get_mean(y[(j - 1) * m: j * m])
            g_approx = get_mean(get_g_x_value(x[(j - 1) * m: j * m]))
            bias_square += np.sum(get_square_diff(g_approx, get_g_x_value(x[(j - 1) * m: j * m])))
            variance += np.sum(get_square_diff(c_j, g_approx))
            empirical_error += np.sum(get_square_diff(c_j, get_g_x_value(x[(j - 1) * m: j * m])))

        bias_square /= n
        variance /= n
        empirical_error /= n

        empirical_err.append(empirical_error)
        bias_square_err.append(bias_square)
        variance_err.append(variance)
        # summing up bias and variance for total error
        total_err.append(bias_square + variance)

    plt.figure(figsize=(10, 6))
    plt.plot(m_vals, empirical_err, label='Avg Empirical Error', marker='x')
    plt.plot(m_vals, bias_square_err, label='Avg Bias-Square', marker='x')
    plt.plot(m_vals, variance_err, label='Avg Variance', marker='x')
    plt.plot(m_vals, total_err, label='Total Avg Error (Bias + Variance)', marker='x')

    plt.xlabel('m values')
    plt.ylabel('Error Values')
    plt.title('Errors vs m values')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    n = 256
    sigma_square = 0.5
    m_vals = [1, 2, 4, 8, 16, 32]
    x = np.arange(1, n + 1) / n

    err_plotter(n=n, x=x, sigma_square=sigma_square, m_vals=m_vals)