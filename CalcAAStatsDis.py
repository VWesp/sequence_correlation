import os
import sys
import numpy as np
import sympy as sp
import pandas as pd
import seaborn as sns
import collections as col
import scipy.stats as sci
import equation_functions as ef
from Bio.Data import CodonTable
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# Function for sorting and returning the amino acids based on a genetic code
def get_amino_acids(id):
    codon_table = CodonTable.unambiguous_dna_by_id[id]
    codon_map = col.Counter(codon_table.forward_table.values())
    codon_map = dict(sorted(codon_map.items(), key=lambda x: (x[1], x[0])))
    return list(codon_map.keys())


if __name__ == "__main__":
    path, output = sys.argv[1:3]
    # Load amino acid distribution file
    aa_dis_df = pd.read_csv(path, sep="\t", index_col=0, dtype=str)
    aa_dis_df["GC"] = aa_dis_df["GC"].astype(float)
    aa_dis_df["Length"] = aa_dis_df["Length"].astype(int)

    # Get the amino acid order based on the Standard genetic code
    amino_acids = get_amino_acids(1)

    # IDs of the proteomes
    proteome_ids = set(aa_dis_df["Genome_Tax_ID"])

    # Calculate metabolic cost of amino acids based on Akashai and Gojobori
    # and convert them to percentages
    costs = {"A": 11.7, "C": 24.7, "D": 12.7, "E": 15.3, "F": 52.0,
             "G": 11.7, "H": 38.3, "I": 32.3, "K": 30.3, "L": 27.3,
             "M": 34.3, "N": 14.7, "P": 20.3, "Q": 16.3, "R": 27.3,
             "S": 11.7, "T": 18.7, "V": 23.3, "W": 74.3, "Y": 50.0}
    costs_inv = {aa:1/costs[aa] for aa in amino_acids}
    costs_inv_adj = {aa:costs_inv[aa]/np.sum(list(costs_inv.values()))
                     for aa in amino_acids}

    # Initialize list of amino acids and iterate over genetic codes
    for code_id, codon_table in CodonTable.unambiguous_dna_by_id.items():
        # Name of the genetic code
        genetic_name = " ".join(codon_table.names[:-1]).lower().replace(" ","_")
        plot_name = genetic_name.capitalize().replace("_", " ")
        print(f"Current code: {plot_name}...")

        code_output = os.path.join(output, genetic_name)
        os.makedirs(code_output, exist_ok=True)

        # Map amino acids to their corresponding codons
        codon_map = col.defaultdict(list)
        for codon, aa in codon_table.forward_table.items():
            codon_map[aa].append(codon)

        # Calculate the frequency percentage for each amino acid based on the
        # codon number
        total_codon_num = np.sum([len(codon_map[aa]) for aa in amino_acids])
        codon_num_adj = {aa:len(codon_map[aa])/total_codon_num
                         for aa in amino_acids}


        # Load frequency functions for each amino acid based on GC content
        g = sp.symbols("g", float=True)
        freq_funcs = ef.build_functions(codon_map)["amino"]

        a = 0.2
        # Metabolic amino acid cost adjusted by codon number
        cod_costs = {aa:(1-a)*costs[aa]+a/codon_num_adj[aa]
                     for aa in amino_acids}
        cod_costs_inv = {aa:1/cod_costs[aa] for aa in amino_acids}
        cod_costs_inv_adj = {aa:(cod_costs_inv[aa]
                                 / np.sum(list(cod_costs_inv.values())))
                             for aa in amino_acids}

        gen_code_l = []
        # Loop over all proteomes and calculate the observed and theoretical
        # amino acid frequencies
        for id in proteome_ids:
            proteome_df = aa_dis_df[aa_dis_df["Genome_Tax_ID"]==id]
            proteome_df = proteome_df[amino_acids+["GC", "Length"]]
            # Count for each protein the amino acids based on the observed
            # codons
            for aa in amino_acids:
                proteome_df[aa] = proteome_df[aa].apply(lambda row: sum(
                                                        int(codon.split(":")[1])
                                               for codon in row.split(";")
                                               )
                                   )

            # Sum the observed amino acids for each protein
            proteome_df["AA_Sum"] = proteome_df[amino_acids].sum(axis=1)
            # Filter out all proteins where the number of observed amino acids
            # does not equal the length of he protein
            proteome_df = proteome_df[proteome_df["AA_Sum"]==proteome_df["Length"]]
            proteome_df = proteome_df.drop("AA_Sum", axis=1)
            # Mean values over the entire dataframe
            proteome_df = proteome_df.mean(axis=0)

            # Calculated amino acid frequencies based on GC content
            gc_freqs = {aa:float(freq_funcs[aa].subs(g, proteome_df["GC"]))
                        for aa in amino_acids}

            # Metabolic amino acid cost adjusted by GC content
            gc_costs = {aa:(1-a)*costs[aa]+a/gc_freqs[aa] for aa in amino_acids}
            gc_costs_inv = {aa:1/gc_costs[aa] for aa in amino_acids}
            gc_costs_inv_adj = {aa:(gc_costs_inv[aa]
                                    / np.sum(list(gc_costs_inv.values())))
                                for aa in amino_acids}

            for aa in amino_acids:
                proteome_df[aa+"_codon"] = (codon_num_adj[aa] *
                                                          proteome_df["Length"])
                proteome_df[aa+"_gc"] = gc_freqs[aa] * proteome_df["Length"]
                proteome_df[aa+"_cost"] = (costs_inv_adj[aa] *
                                                          proteome_df["Length"])
                proteome_df[aa+"_cost_codon"] = (cod_costs_inv_adj[aa] *
                                                          proteome_df["Length"])
                proteome_df[aa+"_cost_gc"] = (gc_costs_inv_adj[aa] *
                                                          proteome_df["Length"])

            gen_code_l.append(proteome_df)

        gen_code_df = pd.DataFrame(gen_code_l)
        gen_code_df.index = proteome_ids
        gen_code_df.index.name = "Proteome_IDs"
        gen_code_df = gen_code_df.dropna()

        # Calculate all Pearson and Spearman correlation coefficients
        for aa_type in ["codon", "gc", "cost", "cost_codon", "cost_gc"]:
            aa_ar = [aa+f"_{aa_type}" for aa in amino_acids]
            # Calculate Pearson correlation between the observed frequencies
            # calculated frequencies for each type
            df_ar = [f"pearson {aa_type}", f"p-pearson {aa_type}"]
            gen_code_df[df_ar] = gen_code_df.apply(lambda row:  pd.Series(
                                    sci.pearsonr(row[amino_acids], row[aa_ar])),
                                    axis=1
                                 )
            # Calculate Spearman correlation between the observed frequencies
            # calculated frequencies for each type
            df_ar = [f"spearman {aa_type}", f"p-spearman {aa_type}"]
            gen_code_df[df_ar] = gen_code_df.apply(lambda row:  pd.Series(
                                    sci.spearmanr(row[amino_acids], row[aa_ar])),
                                    axis=1
                                 )

        gen_code_df.to_csv(os.path.join(code_output, "proteome_freqs.csv"),
                           sep="\t")
