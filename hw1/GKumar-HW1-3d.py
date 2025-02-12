import numpy as np
import matplotlib.pyplot as plt

def estimator_fn(mu, sigma, c_val):
    n = np.arange(10, 1500, 20)
    expt_count = 1000
    estimates = []

    for sample_count in n:
        sample_list = np.random.normal(loc=mu, scale=sigma, size=(expt_count, sample_count))
        sample_mean = np.mean(sample_list, axis=1)
        count = 0
        for x in sample_mean:
            count += 1 if x > c_val else 0
        estimates.append(count/expt_count)

    # arrived at after plugging in mu and sigma values
    chebyshev = 1/n

    plt.figure(figsize=(10, 6))
    plt.plot(n, estimates, label=f"Empirical P[Mn(X) ≥ {c_val}]")
    plt.plot(n, chebyshev, label='Chebyshev Bound', linestyle='--')
    plt.xlabel('n (Number of Samples)')
    plt.ylabel('Probability')
    plt.title(f'Graph depicting Empirical Probability and associated Chebyshev Bound for Gaussian Distribution (µ={mu}, σ={sigma})')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    mu = 0
    sigma = 3
    estimator_fn(mu, sigma, mu + sigma)