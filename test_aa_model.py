import sys
import numpy as np
import sympy as sp
import pandas as pd
import collections as col
import equation_functions as ef
from Bio.Data import CodonTable
from scipy.optimize import minimize
from sklearn.model_selection import KFold


def objective(params, P_values, B_values, C_values, observed):
    a,b,e = params
    sse = 0
    # Loop over all proteins
    for i in range(len(observed)):
        denom_norm = sum(
            (a * P_values[i][j] + (1 - a) * B_values[j]) / C_values[j]**b
            for j in range(len(observed[i]))
        )

        # Loop over all amino_acids
        for j in range(len(observed[i])):
            prediction = ((a * P_values[i][j] + (1 - a) * B_values[j]) /
                         (C_values[j]**b * denom_norm)) + e
            error = observed[i][j] - prediction
            sse += error**2

    return sse


if __name__ == "__main__":
    print("Reading file...")
    df = pd.read_csv(sys.argv[1], sep="\t")
    codon_table = CodonTable.unambiguous_dna_by_id[1]
    genetic_name = " ".join(codon_table.names[:-1]).lower().replace(" ", "_")

    # Map amino acids to their corresponding codons
    aa_to_codon = col.defaultdict(list)
    for codon, aa in codon_table.forward_table.items():
        aa_to_codon[aa].append(codon)

    # Count the number of codons for each amino acid
    codon_count = col.Counter(codon_table.forward_table.values())
    amino_acids = sorted(codon_count, key=lambda x: (codon_count[x], x))
    aa_to_codon = {aa:aa_to_codon[aa] for aa in amino_acids}
    baselines = [codon_count[aa]/sum(codon_count.values())
                 for aa in amino_acids]

    print("Calculating frequencies...")
    observed_freqs = []
    g = sp.symbols("g", float=True)
    freq_funcs = ef.build_functions(aa_to_codon)["amino"]
    calculated_freqs = []
    for index,row in df.iterrows():
        aa_count = {aa:sum(int(c.split(":")[1]) for c in row[aa].split(";")
                        if c.split(":")[0] != "XXX")
                    for aa in amino_acids}
        aa_obs_freqs = [aa_count[aa]/sum(list(aa_count.values()))
                        for aa in amino_acids]
        observed_freqs.append(aa_obs_freqs)

        aa_calc_freqs = [float(freq_funcs[aa].subs(g, row["GC"]))
                         for aa in amino_acids]
        calculated_freqs.append(aa_calc_freqs)

    costs = {"A": 11.7, "C": 24.7, "D": 12.7, "E": 15.3, "F": 52.0, "G": 11.7,
             "H": 38.3, "I": 32.3, "K": 30.3, "L": 27.3, "M": 34.3, "N": 14.7,
             "P": 20.3, "Q": 16.3, "R": 27.3, "S": 11.7, "T": 18.7, "V": 23.3,
             "W": 74.3, "Y": 50.0}
    costs = [costs[aa] for aa in amino_acids]

    init = [0.5, 1, 0.1]
    kf = KFold(n_splits=10, shuffle=True)
    opt_parameters = []
    fold_errors = []
    k = 1
    for train_index, test_index in kf.split(observed_freqs):
        print("\r -> Current fold: {}".format(k), end="")
        # Split the data into training and testing sets
        train_calculated_freqs = [calculated_freqs[i] for i in train_index]
        train_observed_freqs = [observed_freqs[i] for i in train_index]
        test_calculated_freqs = [calculated_freqs[i] for i in test_index]
        test_observed_freqs = [observed_freqs[i] for i in test_index]

        results = minimize(objective, init, args=(train_calculated_freqs,
                                        baselines, costs, train_observed_freqs))
        a_opt,b_opt,e_opt = results.x
        opt_parameters.append([a_opt,b_opt,e_opt])
        test_sse = objective([a_opt, b_opt, e_opt], test_calculated_freqs,
                              baselines, costs, test_observed_freqs)
        fold_errors.append(test_sse)
        k += 1

    print()
    print(fold_errors)
    error_ave = np.average(fold_errors)
    print("Average Cross-Validation SSE: {}".format(error_ave))
    error_std = np.std(fold_errors)
    print("Std Cross-Validation SSE: {}".format(error_std))
