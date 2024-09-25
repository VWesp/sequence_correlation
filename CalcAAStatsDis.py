import os
import sys
import yaml
import shutil
import numpy as np
import sympy as sp
import pandas as pd
import seaborn as sns
import collections as col
import scipy.stats as sci
import multiprocessing as mp
from functools import partial
import equation_functions as ef
from Bio.Data import CodonTable
import matplotlib.pyplot as plt
from settings import AA_TO_ONE_LETTER


import warnings
warnings.filterwarnings("ignore")


def calcCostFreq(df):
    df = df.copy()
    # Replace all values with 0 with 1 for amino acids that didn't appear
    df.replace(0, 1, inplace=True)
    # Calculate the negative natural logarithm of each amino acid frequency
    aa_ln = np.log(df)
    # Estimate constant C
    const_c = -np.mean(aa_ln, axis=1)
    # Calculate energetic cost
    en_cost = (-aa_ln).subtract(const_c, axis=0)
    # Calculate the exponentials of the negative energetic costs
    exp_neg_cost = np.exp(-en_cost)
    # Sum the exponentials
    sum_exp = np.sum(exp_neg_cost, axis=1)
    # Return the frequency of each amino acid
    return exp_neg_cost.div(sum_exp, axis=0)


def calcCodeStats(path_code, path_dis, output, progress_dict):
    # Name of the genetic code
    genetic_name = os.path.basename(path_code).split(".")[0]

    code_output = os.path.join(output, genetic_name)
    os.makedirs(code_output, exist_ok=True)

    # Get the canonical amino acid order
    amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
                   "A", "G", "P", "T", "V", "L", "R", "S"]
    aas_cod = [f"{aa}_cod" for aa in amino_acids]
    aas_gc = [f"{aa}_gc" for aa in amino_acids]
    aas_ener_cod = [f"{aa}_ener_cod" for aa in amino_acids]
    aas_ener_gc = [f"{aa}_ener_gc" for aa in amino_acids]

    # Load genetic code file
    yaml_code = None
    with open(path_code, "r") as code_reader:
        yaml_code = yaml.safe_load(code_reader)

    # Map amino acids to their one letter code
    codon_map = {AA_TO_ONE_LETTER[aa]:codons
                 for aa,codons in yaml_code.items()}
    codon_map = {aa:codon_map[aa] for aa in amino_acids}

    # Calculate the frequency percentage for each amino acid based on the
    # codon number
    total_codon_num = np.sum([len(codon_map[aa]) for aa in amino_acids])
    codon_num_adj = {aa:len(codon_map[aa])/total_codon_num
                     for aa in amino_acids}

    # Load frequency functions for each amino acid based on the codons and
    # GC content
    g = sp.symbols("g", float=True)
    freq_funcs = ef.build_functions(codon_map)

    cor_types = {"cod": aas_cod, "gc": aas_gc, "ener_cod": aas_ener_cod,
                 "ener_gc": aas_ener_gc}

    # Load amino acid distribution file
    dis_df = pd.read_csv(path_dis, sep="\t", index_col=0, dtype=str)
    dis_df["GC"] = dis_df["GC"].astype(float)
    dis_df["Length"] = dis_df["Length"].astype(int)

    # Get only the canonical amino acids
    dis_df = dis_df[["Genome_Tax_ID", "Length", "GC"]+amino_acids]
    # Count for each protein the amino acids based on the observed
    # codons
    for aa in amino_acids:
        dis_df[aa] = dis_df[aa].apply(lambda row: sum(int(codon.split(":")[1])
                                                  for codon in row.split(";")))

    # Sum the observed amino acids for each protein
    dis_df["AA_Sum"] = dis_df[amino_acids].sum(axis=1)
    # Filter out all proteins where the number of observed amino acids
    # does not equal the length of he protein
    # Meaning we only look at proteins without non-canonical amino acids
    dis_df = dis_df[dis_df["AA_Sum"]==dis_df["Length"]]
    dis_df = dis_df.drop("AA_Sum", axis=1)
    # Calculate relative abundance of each amino acid for each protein
    dis_df[amino_acids] = dis_df[amino_acids].div(dis_df["Length"], axis=0)

    # Dataframe with the mean values of the input dataframe
    prot_dis_df = dis_df.groupby("Genome_Tax_ID").mean()

    total_prog = (2 + 2*len(amino_acids) + len(prot_dis_df)
                    + len(cor_types)*len(prot_dis_df))

    # Calculate the relative amino acid abundance based on the codon number
    for aa in amino_acids:
        prot_dis_df[f"{aa}_cod"] = codon_num_adj[aa]
        progress_dict[genetic_name] += 1 / total_prog * 100

    # Calculate the frequency of each amino acid based on the codon number and
    # mean GC content of each proteome
    freqs_list = []
    for index,gc in prot_dis_df["GC"].items():
        freqs = list(ef.calculate_frequencies(freq_funcs, gc)["amino"].values())
        freqs_list.append(freqs)
        progress_dict[genetic_name] += 1 / total_prog * 100

    prot_dis_df[aas_gc] = freqs_list

    # Calculate the frequency of each amino acid based on the energetic cost of
    # each amino acidgiven the relative amino acid abundance based on the codon
    # number
    prot_dis_df[aas_ener_cod] = calcCostFreq(prot_dis_df[aas_cod])
    progress_dict[genetic_name] += 1 / total_prog * 100

    # Calculate the frequency of each amino acid based on the energetic cost of
    # each amino acid given the relative amino acid abundance based on the codon
    # number and mean GC content of each proteome
    prot_dis_df[aas_ener_gc] = calcCostFreq(prot_dis_df[aas_gc])
    progress_dict[genetic_name] += 1 / total_prog * 100

    # Calculate all Pearson and Spearman correlation coefficients for factor
    for cor_type,cols in cor_types.items():
        pear_cols = [f"pearson_{cor_type}", f"p_pearson_{cor_type}"]
        pearson_results = []
        spear_cols = [f"spearman_{cor_type}", f"p_spearman_{cor_type}"]
        spearman_results = []
        # Loop over each row of the dataframe
        for i,row in prot_dis_df.iterrows():
            # Pearson correlation
            pearson_cor,pearson_p = sci.pearsonr(row[amino_acids], row[cols])
            pearson_results.append((pearson_cor, pearson_p))
            progress_dict[genetic_name] += 0.5 / total_prog * 100

            # Spearman correlation
            spearman_cor,spearman_p = sci.spearmanr(row[amino_acids],
                                                    row[cols])
            spearman_results.append((spearman_cor, spearman_p))
            progress_dict[genetic_name] += 0.5 / total_prog * 100

        # Assign results to new columns
        prot_dis_df[pear_cols[0]], prot_dis_df[pear_cols[1]] = zip(*pearson_results)
        prot_dis_df[spear_cols[0]], prot_dis_df[spear_cols[1]] = zip(*spearman_results)

    prot_dis_df.to_csv(os.path.join(code_output, "proteome_cor_data.csv"),
                           sep="\t")

    # Calculate the correlations for each amino acid
    correlations = []
    # Loop over all amino acids
    for aa in amino_acids:
        correlations.append({
            "Amino_Acid": aa,
            "Factor": "Amino acid",
            "Mean": np.mean(prot_dis_df[aa]),
            "Pearson_Correlation": None,
            "Pearson_p_value": None,
            "Spearman_Correlation": None,
            "Spearman_p_value": None
        })

        # Calculate Pearson and Spearman correlation coefficients given the
        # mean length of each proteome
        pear_len,p_pear_len = sci.pearsonr(prot_dis_df[aa],
                                           prot_dis_df["Length"])
        spear_len,p_spear_len = sci.spearmanr(prot_dis_df[aa],
                                              prot_dis_df["Length"])
        correlations.append({
            "Amino_Acid": aa,
            "Factor": "Length",
            "Mean": np.mean(prot_dis_df["Length"]),
            "Pearson_Correlation": pear_len,
            "Pearson_p_value": p_pear_len,
            "Spearman_Correlation": spear_len,
            "Spearman_p_value": p_spear_len
        })

        # Calculate Pearson and Spearman correlation coefficients given the
        # mean GC content of each proteome
        pear_gc,p_pear_gc = sci.pearsonr(prot_dis_df[aa], prot_dis_df["GC"])
        spear_gc,p_spear_gc = sci.spearmanr(prot_dis_df[aa], prot_dis_df["GC"])
        correlations.append({
            "Amino_Acid": aa,
            "Factor": "GC",
            "Mean": np.mean(prot_dis_df["GC"]),
            "Pearson_Correlation": pear_gc,
            "Pearson_p_value": p_pear_gc,
            "Spearman_Correlation": spear_gc,
            "Spearman_p_value": p_spear_gc
        })

        # Calculate Pearson and Spearman correlation coefficients given the
        # the codon number and mean GC content of each proteome
        pear_codon_gc,p_pear_codon_gc = sci.pearsonr(prot_dis_df[aa],
                                                     prot_dis_df[f"{aa}_gc"])
        spear_codon_gc,p_spear_codon_gc = sci.spearmanr(prot_dis_df[aa],
                                                        prot_dis_df[f"{aa}_gc"])
        correlations.append({
            "Amino_Acid": aa,
            "Factor": "Codon+GC",
            "Mean": np.mean(prot_dis_df[f"{aa}_gc"]),
            "Pearson_Correlation": pear_codon_gc,
            "Pearson_p_value": p_pear_codon_gc,
            "Spearman_Correlation": spear_codon_gc,
            "Spearman_p_value": p_spear_codon_gc
        })

        # Calculate Pearson and Spearman correlation coefficients given the
        # energetic cost of each amino acid based on the relative amino acid
        # abundance based on the codon number and meanGC content of each
        # proteome
        pear_ener_gc,p_pear_ener_gc = sci.pearsonr(prot_dis_df[aa],
                                                   prot_dis_df[f"{aa}_ener_gc"])
        spear_ener_gc,p_spear_ener_gc = sci.spearmanr(prot_dis_df[aa],
                                                      prot_dis_df[f"{aa}_ener_gc"])
        correlations.append({
            "Amino_Acid": aa,
            "Factor": "Codon+GC+Cost",
            "Mean": np.mean(prot_dis_df[f"{aa}_ener_gc"]),
            "Pearson_Correlation": pear_ener_gc,
            "Pearson_p_value": p_pear_ener_gc,
            "Spearman_Correlation": spear_ener_gc,
            "Spearman_p_value": p_spear_ener_gc
        })

        progress_dict[genetic_name] += 1 / total_prog * 100

    amino_acid_df = pd.DataFrame(correlations)
    amino_acid_df.to_csv(os.path.join(code_output,
                                      "amino_acid_cor_data.csv"),
                         sep="\t")

    progress_dict[genetic_name] = 100


# Function to display progress of all workers on individual lines
def progress_report(progress_dict, processes):
    # Get terminal height and width
    terminal_width = shutil.get_terminal_size().columns

    # Print initial lines for all processes
    for name in processes:
        print(f"{name}: 0.00%")

    while(any(progress_dict[name] < 100 for name in processes)):
        # Move the cursor back up to the first process line
        sys.stdout.write(f"\033[{len(processes)}F")

        # Print updated progress for each process
        for name in processes:
            progress_value = progress_dict.get(name, 0)
            progress_str = f"{name}: {progress_value:.2f}%"

            # Truncate progress string if it exceeds terminal width
            if(len(progress_str) > terminal_width):
                progress_str = progress_str[:terminal_width - 3] + "..."

            print(progress_str)

    # Move the cursor down after all processes are completed
    sys.stdout.write(f"\033[{len(processes)}E")
    print("All processes completed.")


if __name__ == "__main__":
    path_dis,code_path,output,procs = sys.argv[1:5]
    os.makedirs(output, exist_ok=True)

    # List of all genetic codes in the folder
    code_paths = [os.path.join(code_path, path)
                  for path in os.listdir(code_path)]

    # use Manager to create a Queue that can be shared between processes
    manager = mp.Manager()
    progress_queue = manager.Queue()

    prog_processes = [os.path.basename(path).split(".")[0]
                      for path in code_paths]

    progress_dict = manager.dict({name:0 for name in prog_processes})

    # Create a pool with a limited number of concurrent workers
    pool = mp.Pool(int(procs))

    # Start a process to report progress
    progress_reporter = mp.Process(target=progress_report, args=(progress_dict,
                                                                prog_processes))

    # Start the progress reporter
    progress_reporter.start()

    # Use the pool to process the tasks concurrently
    for path_code in code_paths:
        pool.apply_async(calcCodeStats, args=(path_code, path_dis, output,
                                              progress_dict))

    # Close the pool and wait for all tasks to finish
    pool.close()
    pool.join()

    # Wait for the progress reporter to finish
    progress_reporter.join()
