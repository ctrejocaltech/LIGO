import numpy as np
from scipy.stats import entropy

# Load the chains from the results files
with open('/Users/ligo/Documents/lisa/petra/results/fake_gaussian_results/relabeled_chains/sorted_fake_gaussian_posterior_0.npy', 'rb') as fp:
    sorted_chains = np.load(fp)

with open('/Users/ligo/Documents/lisa/petra/results/fake_gaussian_results/simulated_chains/fake_gaussian_posterior_0.npy', 'rb') as fp:
    simulated_chains = np.load(fp)

# Check the shape of the loaded chains
print(f"Shape of sorted_chains: {sorted_chains.shape}")
print(f"Shape of simulated_chains: {simulated_chains.shape}")

sorted_chains = sorted_chains.flatten()
simulated_chains = simulated_chains.flatten()

bins = np.linspace(min(sorted_chains.min(), simulated_chains.min()), max(sorted_chains.max(), simulated_chains.max()), 100)
hist_sorted, _ = np.histogram(sorted_chains, bins=bins, density=True)
hist_simulated, _ = np.histogram(simulated_chains, bins=bins, density=True)

hist_sorted = hist_sorted / hist_sorted.sum()
hist_simulated = hist_simulated / hist_simulated.sum()

kl_divergence = entropy(hist_sorted, hist_simulated)

print(f"KL Divergence: {kl_divergence}")
