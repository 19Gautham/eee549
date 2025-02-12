import numpy as np
import matplotlib.pyplot as plt

def step_func(y_vals):
    steps = n // m
    res = [0] * n
    for j in range(steps):
        start = j * m
        end = (j + 1) * m
        # taking the mean of all values that lie within this interval
        c_j = np.sum(y_vals[start:end])/(end-start)
        # assign the step function value to all the values in the interval
        res[start:end] = [c_j] * m
    return res

def get_square_diff(x, y):
    return (x-y) ** 2

def err_plotter_func(n, m):
    # 128 values between [0,1]
    x = np.linspace(0, 1, n)

    # Compute y values
    g1_x = np.sin(2 * np.pi * x)
    g2_x = np.sin(8 * np.pi * x)

    y_g1_approx = step_func(g1_x)
    y_g2_approx = step_func(g2_x)
    # Seeing how the squared error varies
    bias_square_error_g1x = get_square_diff(y_g1_approx, g1_x)
    bias_square_error_g2x = get_square_diff(y_g2_approx, g2_x)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(x, g1_x, label='g1(x) = sin(2πx)')
    plt.plot(x, y_g1_approx, label='Step function estimate of g1(x)')
    plt.plot(x, bias_square_error_g1x, label='Bias-square error', linestyle='--')
    plt.title('Approximation of g1(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, g2_x, label='g2(x) = sin(8πx)')
    plt.plot(x, y_g2_approx, label='Step function estimate of g2(x)', linestyle='--')
    plt.plot(x, bias_square_error_g2x, label='Bias-square error', linestyle='--')
    plt.title('Approximation of g2(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Average bias-square error for g1(x) : {np.mean(bias_square_error_g1x)}")
    print(f"Average bias-square error for g2(x) : {np.mean(bias_square_error_g2x)}")

if __name__ == "__main__":
    n = 256
    m = 128
    err_plotter_func(n, m)