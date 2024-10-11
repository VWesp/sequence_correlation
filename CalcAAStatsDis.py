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
import sklearn.metrics as skl
import equation_functions as ef
from Bio.Data import CodonTable
import matplotlib.pyplot as plt


# One letter code for the amino acids of the genetic codes
AA_TO_ONE_LETTER = {"Methionine": "M", "Threonine": "T", "Asparagine": "N",
                    "Lysine": "K", "Serine": "S", "Arginine": "R",
                    "Valine": "V", "Alanine": "A", "Aspartic_acid": "D",
                    "Glutamic_acid": "E", "Glycine": "G", "Phenylalanine": "F",
                    "Leucine": "L", "Tyrosine": "Y", "Cysteine": "C",
                    "Tryptophane": "W", "Proline": "P", "Histidine": "H",
                    "Glutamine": "Q", "Isoleucine": "I", "Stop": "*"}


def combine_data(paths, amino_acids):
    comb_df = pd.DataFrame()
    for prot in prot_dir:
        df = pd.read_csv(prot, sep="\t", header=0, index_col=0)
        df = df.drop(["Status"], axis=1)
        df_med = df.median(axis=0)
        df_med.name = os.path.basename(prot).split("_")[0]
        comb_df = pd.concat([comb_df, df_med.to_frame().T])

    comb_df = comb_df.fillna(0.0)
    if("X" in comb_df.columns):
        comb_df.loc[:, amino_acids] = comb_df.loc[:, amino_acids].add(comb_df["X"]/20, axis=0)
        comb_df = comb_df.drop(["X"], axis=1)

    if("B" in comb_df.columns):
        comb_df.loc[:, ["N", "D"]] = comb_df.loc[:, ["N", "D"]].add(comb_df["B"]/2, axis=0)
        comb_df = comb_df.drop(["B"], axis=1)

    if("Z" in comb_df.columns):
        comb_df.loc[:, ["Q", "E"]] = comb_df.loc[:, ["Q", "E"]].add(comb_df["Z"]/2, axis=0)
        comb_df = comb_df.drop(["Z"], axis=1)

    if("O" in comb_df.columns):
        comb_df.loc[:, ["K"]] = comb_df.loc[:, ["K"]].add(comb_df["O"], axis=0)
        comb_df = comb_df.drop(["O"], axis=1)

    if("U" in comb_df.columns):
        comb_df.loc[:, ["C"]] = comb_df.loc[:, ["C"]].add(comb_df["U"], axis=0)
        comb_df = comb_df.drop(["U"], axis=1)

    comb_df.index.name = "Proteome_ID"
    comb_df = comb_df[["GC", "Length"]+amino_acids]
    return comb_df


def log(data):
    if(type(data) is float):
        data = data if data > 0 else 1
    else:
        data[data==0] = 1

    return np.log(data)


if __name__ == "__main__":
    prots,codes,output = sys.argv[1:4]

    os.makedirs(output, exist_ok=True)

    prot_dir = [os.path.join(prots, file) for file in os.listdir(prots)]
    code_dir = [os.path.join(codes, file) for file in os.listdir(codes)]
    # Canonical amino acids
    amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
                   "A", "G", "P", "T", "V", "L", "R", "S"]

    df_prot = combine_data(prot_dir, amino_acids)
    df_prot[amino_acids] = df_prot[amino_acids].div(df_prot[amino_acids].sum(axis=1),
                                                                             axis=0)
    entropies = []
    for i,row in df_prot.iterrows():
        entropy = -sum([row[aa] * np.log2(row[aa]) if row[aa] > 0 else 0
                        for aa in amino_acids])
        entropies.append(entropy)

    df_prot["shannon_entropy"] = entropies
    df_prot.to_csv(os.path.join(output, "proteome_data.csv"), sep="\t")

    for code in code_dir:
        code_name = os.path.basename(code).split(".yaml")[0]
        print(f"Current code {code_name}")
        code_output = os.path.join(output, code_name)
        os.makedirs(code_output, exist_ok=True)

        # Load genetic code file
        yaml_code = None
        with open(code, "r") as code_reader:
            yaml_code = yaml.safe_load(code_reader)

        # Map amino acids to their one letter code
        codon_map = {AA_TO_ONE_LETTER[aa]:codons
                     for aa,codons in yaml_code.items()}
        codon_map = {aa:codon_map[aa] for aa in amino_acids}

        # Calculate the frequency percentage for each amino acid based on the
        # codon number
        total_codon_num = np.sum([len(codon_map[aa]) for aa in amino_acids])
        codon_num_norm = {aa:len(codon_map[aa])/total_codon_num
                          for aa in amino_acids}

        df_code = pd.DataFrame.from_dict(codon_num_norm, orient="index")
        df_code.to_csv(os.path.join(code_output, "norm_code_data.csv"),
                       sep="\t", header=["frequency"])

        df_cor = pd.DataFrame()
        #############################################
        # Load frequency functions for each amino acid based on the codons and
        # GC content
        g = sp.symbols("g", float=True)
        freq_funcs = ef.build_functions(codon_map)

        for i,row in df_prot.iterrows():
            row_data = np.array(row[amino_acids])
            log_data = np.log2(row_data, where=row_data!=0)

            freqs = np.array(list(ef.calculate_frequencies(freq_funcs,
                                                  row["GC"])["amino"].values()))
            freqs = freqs / freqs.sum()
            log_freqs = np.log2(freqs, where=freqs!=0)
            df_cor.loc[row.name, amino_acids] = freqs
            ##############
            cor, p_cor = sci.spearmanr(row_data, df_code)
            mse = skl.mean_squared_error(log_data, np.log2(df_code))
            df_cor.loc[row.name, "codon_cor"] = cor
            df_cor.loc[row.name, "p_codon_cor"] = p_cor
            df_cor.loc[row.name, "codon_log_mse"] = mse
            ############################
            cor, p_cor = sci.spearmanr(row[amino_acids], freqs)
            mse = skl.mean_squared_error(log_data, log_freqs)
            entropy = -sum([freqs[i] * np.log2(freqs[i]) if freqs[i] > 0 else 0
                            for i in range(len(freqs))])
            df_cor.loc[row.name, "gc_cor"] = cor
            df_cor.loc[row.name, "p_gc_cor"] = p_cor
            df_cor.loc[row.name, "gc_log_mse"] = mse
            df_cor.loc[row.name, "shannon_entropy"] = entropy

        df_cor.to_csv(os.path.join(code_output, "cor_data.csv"), sep="\t")

    print()
