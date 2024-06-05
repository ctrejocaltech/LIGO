import numpy as np
from scipy.stats import entropy

def kl_div(mu1, mu2, cov1, cov2):
    cov2_inv = np.linalg.inv(cov2)
    return 0.5 * (np.trace(np.dot(cov2_inv, cov1)) + np.dot(np.dot((mu2 - mu1).T, cov2_inv), (mu2 - mu1)) - 2 + np.log(np.linalg.det(cov2) / np.linalg.det(cov1)))


def kl_function(simulated_filepath, sorted_filepath):
    with open(simulated_filepath, 'rb') as fp:
        simulated_chains = np.load(fp)
    with open(sorted_filepath, 'rb') as fp:
        sorted_chains = np.load(fp)

    mean_simulated = np.mean(simulated_chains, axis=0)
    mean_sorted = np.mean(sorted_chains, axis=0)

    cov_sim = [np.cov(simulated_chains [:,i,:].T) for i in range(2)]
    cov_sorted = [np.cov(sorted_chains [:,i,:].T) for i in range(2)]    

    mu1 = mean_simulated[0, :]
    cov1 = cov_sim[0]

    mu2 = mean_sorted[0, :]
    cov2 = cov_sorted[0]

    kl_list = [kl_div(mu1, mu2, cov1, cov2)]
    mu1 = mean_simulated[1, :]
    cov1 = cov_sim[1]

    mu2 = mean_sorted[1, :]
    cov2 = cov_sorted[1]

    kl_list.append(kl_div(mu1, mu2, cov1, cov2))

    return(kl_list)

sorted_filepath = '/Users/ligo/Documents/lisa/petra/results/fake_gaussian_results_no_guess/relabeled_chains/sorted_fake_gaussian_posterior_10.npy'
simulated_filepath = '/Users/ligo/Documents/lisa/petra/results/fake_gaussian_results_no_guess/simulated_chains/fake_gaussian_posterior_10.npy'

print(kl_function(simulated_filepath, sorted_filepath))