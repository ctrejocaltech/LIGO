import numpy as np
from scipy.stats import entropy
import pandas as pd
import os
import re

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

    cov_sim = [np.cov(simulated_chains[:,i,:].T) for i in range(2)]
    cov_sorted = [np.cov(sorted_chains[:,i,:].T) for i in range(2)]    

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

    return kl_list, simulated_chains, sorted_chains

def find_npy_files(base_dir):
    npy_files = []
    for root, dirs, files in os.walk(base_dir):
        npy_files.extend(
            os.path.join(root, file) for file in files if file.endswith('.npy')
        )
    return npy_files

def extract_number(filename):
    if match := re.search(r'_(\d+)\.npy$', filename):
        return int(match[1])
    return None

def pair_files(npy_files):
    paired_files = {}
    for file in npy_files:
        number = extract_number(os.path.basename(file))
        if number is not None:
            if number not in paired_files:
                paired_files[number] = {}
            if 'sorted' in file:
                paired_files[number]['sorted'] = file
            else:
                paired_files[number]['simulated'] = file
    return paired_files

base_dir = '/Users/ligo/Documents/lisa/petra/results/fake_gaussian_results_no_guess'
output_filepath = '/Users/ligo/Documents/lisa/petra/results/kl_divergence_results.xlsx'

# Find all .npy files in the directory and subdirectories
npy_files = find_npy_files(base_dir)

# Pair the files based on the number at the end of the filename
paired_files = pair_files(npy_files)

results = []

for number, files in paired_files.items():
    if 'simulated' in files and 'sorted' in files:
        simulated_filepath = files['simulated']
        sorted_filepath = files['sorted']

        kl_results, simulated_chains, sorted_chains = kl_function(simulated_filepath, sorted_filepath)

        results.extend(
            {
                "Simulated File": simulated_filepath,
                "Sorted File": sorted_filepath,
                "KL Divergence": kl_result,                
                "Simulated Chains": simulated_chains[j].flatten().tolist(),
                "Sorted Chains": sorted_chains[j].flatten().tolist(),
            }
            for j, kl_result in enumerate(kl_results)
        )
# Convert the results to a DataFrame
df = pd.DataFrame(results)

# Save the DataFrame to an Excel file
df.to_excel(output_filepath, index=False)

print(f"KL divergence results saved to {output_filepath}")
