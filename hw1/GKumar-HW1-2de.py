import numpy as np
import matplotlib.pyplot as plt

def estimator_fn(c_vals):
    # theta=0.4, map's x_values to probability values
    pmf = {0: 0.3, 6: 0.3, 3: 0.4}
    # Choosing these particular n values
    n = np.arange(10, 200, 10)
    expt_count = 1000
    # print(n)

    pmf_vals = list(pmf.keys())
    pmf_prob = list(pmf.values())

    # a list of empirical values for each c value
    estimates_dict = {c: [] for c in c_values}

    for sample_count in n:
        # choose values
        sample_list = np.random.choice(a=pmf_vals, size=(expt_count, sample_count), p=pmf_prob)
        sample_mean = np.mean(sample_list, axis=1)
        for c in c_vals:
            count = 0
            for x in sample_mean:
                count += 1 if x > c else 0
            estimates_dict[c].append(count / expt_count)

    for c in c_vals:
        # does not depend on the number of the samples
        markov = 3/c
        # Chebyshev
        chebyshev = 9 * (1-0.4) / (((c-3)**2) * n)
        # Hoeffding
        hoeffding = np.exp(-n * (c - 3)**2 / 18)

        plt.figure(figsize=(10, 6))
        plt.plot(n, estimates_dict[c], label=f"Empirical P[Mn(X) >= {c}]", marker='o')
        plt.hlines(markov, xmin=min(n), xmax=max(n), color='r', label='Markov Bound', linestyles='dashed')
        plt.plot(n, chebyshev, label='Chebyshev Bound', marker='s', linestyle='--')
        plt.plot(n, hoeffding, label="Hoeffding's Bound", marker='x', linestyle='--')
        plt.xlabel('n (Number of Samples)')
        plt.ylabel('Probability')
        plt.title(f'Graph depciting Empirical Probability and associated Theoretical Bounds (Markov, Chebyshev, Hoeffding) c = {c}')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    c_values = [4, 4.5, 6]
    estimator_fn(c_values)