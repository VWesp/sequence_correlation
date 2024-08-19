import os
import sys
import numpy as np
import sympy as sp
import pandas as pd
import textwrap as tw
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


# Function to calculate the sum of values
def count_amino_acids(codon_dis, canon_codons):
    return sum(
        int(codon.split(":")[1])
        for codon in codon_dis.split(";")
        if codon.split(":")[0] in canon_codons
    )


if __name__ == "__main__":
    path,ouput,name = sys.argv[1:4]
    # Load amino acid distribution file
    aa_dis_df = pd.read_csv(path, sep="\t", index_col=0, dtype=str)
    aa_dis_df["GC"] = aa_dis_df["GC"].astype(float)
    aa_dis_df["Length"] = aa_dis_df["Length"].astype(int)

    cmap = plt.get_cmap("viridis")

    # Get the amino acid order based on the Standard genetic code
    amino_acids = get_amino_acids(1)
    aa_gc_cols = [aa+"_gc" for aa in amino_acids]

    # IDs of the proteomes
    proteome_ids = set(aa_dis_df["Genome_Tax_ID"])

    # Calculate the average GC content
    ave_gc = np.average(aa_dis_df["GC"])

    # Group amino acids based on their attributes
    aa_groups = {"Aliphatic": ["A", "G", "I", "L", "M", "V"], "Aromatic": ["F",
                 "W", "Y"], "Charged": ["D", "E", "H", "K", "R"],
                 "Uncharged": ["C", "N", "P", "Q", "S", "T"]}

    # Calculate metabolic cost of amino acids based on Akashai and Gojobori
    # and convert them to percentages
    costs = {"A": 11.7, "C": 24.7, "D": 12.7, "E": 15.3, "F": 52.0,
             "G": 11.7, "H": 38.3, "I": 32.3, "K": 30.3, "L": 27.3,
             "M": 34.3, "N": 14.7, "P": 20.3, "Q": 16.3, "R": 27.3,
             "S": 11.7, "T": 18.7, "V": 23.3, "W": 74.3, "Y": 50.0}
    costs_inv = {aa:1/costs[aa] for aa in amino_acids}
    costs_inv_adj = {aa:costs_inv[aa]/np.sum(list(costs_inv.values()))
                     for aa in amino_acids}

    # Calculate ATP yield of amino acids based on Kaleta and convert them to
    # percentages
    atp_yield = {"A": [2,1,0.5], "R": [1,0.5,0.27], "N": [1.73,1,0.47],
                 "D": [2,1,0.5], "C": [1.24,0.71,0.32], "E": [1,0.5,0.33],
                 "Q": [1,0.5,0.33], "G": [2,1,0.5], "H": [0.75,0.42,0.19],
                 "I": [0.79,0.45,0.21], "L": [0.67,0.33,0.2], "P": [1,0.5,0.29],
                 "M": [0.71,0.4,0.19], "F": [0.57,0.3,0.14], "S": [2,1,0.5],
                 "T": [1.37,0.78,0.37], "W": [0.44,0.25,0.11], "V": [1,0.5,0.25],
                 "Y": [0.57,0.3,0.15], "K": [0.84,0.48,0.22]}
    atp_yield_ave = {aa:np.average(atp_yield[aa]) for aa in amino_acids}
    atp_yield_adj ={aa:atp_yield_ave[aa]/np.sum(list(atp_yield_ave.values()))
                    for aa in amino_acids}

    # Initialize list of amino acids and iterate over genetic codes
    for code_id, codon_table in CodonTable.unambiguous_dna_by_id.items():
        # Name of the genetic code for the directory
        genetic_name = " ".join(codon_table.names[:-1]).lower().replace(" ","_")
        # Name of the genetic name for the plot
        plot_name = genetic_name.capitalize().replace("_", " ")
        print(f"Current code: {plot_name}...")

        code_output = os.path.join(ouput, genetic_name)
        os.makedirs(code_output, exist_ok=True)

        # Map amino acids to their corresponding codons
        codon_map = col.defaultdict(list)
        for codon, aa in codon_table.forward_table.items():
            codon_map[aa].append(codon)

        # List of canonical codons in each genetic code
        canon_codons = sorted([codon for codons in list(codon_map.values())
                               for codon in codons])

        # Calculate the frequency percentage for each amino acid based on the
        # codon number
        total_codon_num = np.sum([len(codon_map[aa]) for aa in amino_acids])
        codon_num_adj = {aa:len(codon_map[aa])/total_codon_num
                         for aa in amino_acids}

        # Load frequency functions for each amino acid based on GC content
        g = sp.symbols("g", float=True)
        freq_funcs = ef.build_functions(codon_map)["amino"]

        # Dataframe for amino acid frequency for each proteome based on
        # observation and calculation
        amino_acid_df = pd.DataFrame(index=range(len(proteome_ids)),
                                     columns=amino_acids+["Len_mean", "GC_mean"])
        amino_acid_df.index = proteome_ids
        amino_acid_df.index.name = "Proteome_ID"
        # Loop over all proteomes and calculate the observed and statistical
        # amino acid frequencies
        for id in proteome_ids:
            proteome_df = aa_dis_df[aa_dis_df["Genome_Tax_ID"]==id]
            # Count for each protein the amino acids based on the observed
            # codons
            for aa in amino_acids:
                proteome_df[aa] = proteome_df[aa].apply(count_amino_acids,
                                                      canon_codons=canon_codons)

            # Sum the observed amino acids for each protein
            proteome_df["Sum"] = proteome_df[amino_acids].sum(axis=1)
            # Filter out all proteins where the number of observed amino acids
            # does not equal the length of protein
            proteome_df = proteome_df[proteome_df["Sum"]==proteome_df["Length"]]
            # Calculate the observed amino acids frequencies for the proteome
            amino_acid_df.loc[id, amino_acids] = (proteome_df[amino_acids].sum()
                                                  / proteome_df["Sum"].sum())
            # Calculate the mean length of GC content of all proteins of the
            # proteome
            amino_acid_df.loc[id, "Len_mean"] = np.mean(proteome_df["Length"])
            gc_mean = np.mean(proteome_df["GC"])
            amino_acid_df.loc[id, "GC_mean"] = gc_mean
            # Calculate for the proteome the expected amino acid frequencies
            # based on the mean GC content
            for aa in amino_acids:
                amino_acid_df.loc[id,aa+"_gc"] = float(freq_funcs[aa].subs(g,
                                                                       gc_mean))

        # Dataframes for Pearson and Spearman correlations
        pearson_df = pd.DataFrame(columns=["correlation", "p-value", "Type"])
        spearman_df = pd.DataFrame(columns=["correlation", "p-value", "Type"])
        cor_data = {"Codon number": list(codon_num_adj.values()),
                    "GC content": None,
                    "Metbaolic cost": list(costs_inv_adj.values()),
                    "ATP yield": list(atp_yield_adj.values())}
        # Calculate the correlation coefficients of different factor for each
        # proteome
        for type,data in cor_data.items():
            cor_pear_df = pd.DataFrame(index=range(len(proteome_ids)),
                                     columns=["correlation", "p-value", "Type"])
            cor_pear_df.index = proteome_ids
            cor_pear_df.index.name = "Proteome_ID"
            cor_spear_df = pd.DataFrame(index=range(len(proteome_ids)),
                                     columns=["correlation", "p-value", "Type"])
            cor_spear_df.index = proteome_ids
            cor_spear_df.index.name = "Proteome_ID"
            if(data is None):

        break
                # Calculate Pearson correlation based on GC content
                cor_pear_df[["correlation", "p-value"]] = amino_acid_df.apply(
                                                    lambda row: pd.Series(sci.pearsonr(
                                                    row[amino_acids], row[aa_gc_cols])),
                                                    axis=1)
                # Calculate Spearman correlation based on GC content
                cor_spear_df[["correlation", "p-value"]] = amino_acid_df.apply(
                                                    lambda row: pd.Series(sci.spearmanr(
                                                    row[amino_acids], row[aa_gc_cols])),
                                                    axis=1)
            else:
                # Calculate Pearson correlation based on codon number,
                # metabolic cost or ATP yield
                cor_pear_df[["correlation", "p-value"]] = amino_acid_df.apply(
                                                    lambda row: pd.Series(sci.pearsonr(
                                                    row[amino_acids], data)), axis=1)
                # Calculate Spearman correlation based on codon number,
                # metabolic cost or ATP yield
                cor_spear_df[["correlation", "p-value"]] = amino_acid_df.apply(
                                                    lambda row: pd.Series(sci.spearmanr(
                                                    row[amino_acids], data)), axis=1)

            cor_pear_df["Type"] = [type] * len(cor_pear_df)
            pearson_df = pd.concat([pearson_df, cor_pear_df])
            cor_spear_df["Type"] = [type] * len(cor_spear_df)
            spearman_df = pd.concat([spearman_df, cor_spear_df])

        amino_acid_df.to_csv(f"{code_output}/aa_dis_frequencies.csv", sep="\t")
        pearson_df.to_csv(f"{code_output}/pearsons.csv", sep="\t")
        spearman_df.to_csv(f"{code_output}/spearmans.csv", sep="\t")

        overall_aa_mean = {aa:amino_acid_df[amino_acids].mean()[aa]
                           for aa in amino_acids}
        overall_aa_gc_mean = {aa:amino_acid_df[aa_gc_cols].mean()[aa+"_gc"]
                              for aa in amino_acids}
        for type,aa_list in aa_groups.items():
            fig, ax = plt.subplots()
            x_data = np.arange(len(aa_list))
            w = 0.15

            # Observed frequency of amino acids and plot them
            obs_aas = np.array([overall_aa_mean[aa] for aa in aa_list])
            ax.bar(x_data-w*2, obs_aas, width=w, color=cmap(0.1),
                   edgecolor="black", linewidth=0.75, hatch="//",
                   label="Observed amount")

            # Expected frequency of amino acids based on codon number and plot
            # them
            codon_aas = np.array([codon_num_adj[aa] for aa in aa_list])
            ax.bar(x_data-w, codon_aas, width=w, color=cmap(0.3),
                   edgecolor="black", linewidth=0.75, hatch="\\\\",
                   label="Codon number")
            # Calculate Pearson and Spearman correlation coefficients for the
            # codon number
            codon_pcc, codon_pcc_p = sci.pearsonr(list(overall_aa_mean.values()),
                                                  list(codon_num_adj.values()))
            codon_r2, codon_r2_p = sci.spearmanr(list(overall_aa_mean.values()),
                                                 list(codon_num_adj.values()))
            # Add correlation text box for the codon number
            ax.text(1.02, 0.65, f"$Codon$ correlation:\n"
                    f"  - $pcc$: {codon_pcc:.2f}; $p_{{pcc}}$: {codon_pcc_p:.1e}\n"
                    f"  - $r2$: {codon_r2:.2f}; $p_{{r2}}$: {codon_r2_p:.1e}",
                    transform=ax.transAxes, fontsize=8, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white",
                              edgecolor="grey", alpha=0.5))

            # Expected frequency of amino acids based on GC content and plot
            # them
            gc_aas = np.array([overall_aa_gc_mean[aa] for aa in aa_list])
            ax.bar(x_data, gc_aas, width=w, color=cmap(0.5),
                   edgecolor="black", linewidth=0.75, hatch="//",
                   label="GC content")
            # Calculate Pearson and Spearman correlation coefficients for the
            # GC content
            gc_pcc, gc_pcc_p = sci.pearsonr(list(overall_aa_mean.values()),
                                            list(overall_aa_gc_mean.values()))
            gc_r2, gc_r2_p = sci.spearmanr(list(overall_aa_mean.values()),
                                           list(overall_aa_gc_mean.values()))
            # Add correlation text box for the GC content
            ax.text(1.02, 0.5, f"$GC$ correlation:\n"
                    f"  - $pcc$: {gc_pcc:.2f}; $p_{{pcc}}$: {gc_pcc_p:.1e}\n"
                    f"  - $r2$: {gc_r2:.2f}; $p_{{r2}}$: {gc_r2_p:.1e}",
                    transform=ax.transAxes, fontsize=8, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white",
                              edgecolor="grey", alpha=0.5))

            # Expected frequency of amino acids based on metabolic cost and plot
            # them
            cost_aas = np.array([costs_inv_adj[aa] for aa in aa_list])
            ax.bar(x_data+w, cost_aas, width=w, color=cmap(0.7),
                   edgecolor="black", linewidth=0.75, hatch="\\\\",
                   label="Metabolic cost")
            # Calculate Pearson and Spearman correlation coefficients for the
            # metabolic cost
            cost_pcc, cost_pcc_p = sci.pearsonr(list(overall_aa_mean.values()),
                                                list(costs_inv_adj.values()))
            cost_r2, cost_r2_p = sci.spearmanr(list(overall_aa_mean.values()),
                                               list(costs_inv_adj.values()))
            # Add correlation text box for the metabolic cost
            ax.text(1.02, 0.35, f"$Cost$ correlation:\n"
                    f"   - $pcc$: {cost_pcc:.2f}; $p_{{pcc}}$: {cost_pcc_p:.1e}\n"
                    f"   - $r2$: {cost_r2:.2f}; $p_{{r2}}$: {cost_r2_p:.1e}",
                    transform=ax.transAxes, fontsize=8, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white",
                              edgecolor="grey", alpha=0.5))

            # Expected frequency of amino acids based on ATP yield and plot them
            yield_aas = np.array([atp_yield_adj[aa] for aa in aa_list])
            ax.bar(x_data+w*2, yield_aas, width=w, color=cmap(0.9),
                   edgecolor="black", linewidth=0.75, hatch="//",
                   label="ATP yield")
            # Calculate Pearson and Spearman correlation coefficients for the
            # ATP yield
            cost_pcc, cost_pcc_p = sci.pearsonr(list(overall_aa_mean.values()),
                                                list(atp_yield_adj.values()))
            cost_r2, cost_r2_p = sci.spearmanr(list(overall_aa_mean.values()),
                                               list(atp_yield_adj.values()))
            # Add correlation text box for the metabolic cost
            ax.text(1.02, 0.2, f"$Yield$ correlation:\n"
                    f"   - $pcc$: {cost_pcc:.2f}; $p_{{pcc}}$: {cost_pcc_p:.1e}\n"
                    f"   - $r2$: {cost_r2:.2f}; $p_{{r2}}$: {cost_r2_p:.1e}",
                    transform=ax.transAxes, fontsize=8, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white",
                              edgecolor="grey", alpha=0.5))

            ax.set_xticks(np.arange(len(aa_list)), aa_list)
            ax.set_xlabel(f"{type} amino acid")
            ax.set_ylabel("Mean amino acid count")
            ax.legend(loc="upper center", bbox_to_anchor=(1.19, 1.02),
                      fancybox=True)
            title = tw.fill(f"{name} - Correlation between amino acid "
                            f"and codon number for genetic code: {plot_name}",
                            50)
            plt.title(title)
            for ext in ["svg", "pdf"]:
                plt.savefig(f"{code_output}/{type.lower()}.{ext}",
                            bbox_inches="tight")

            plt.close()

        # Plot the Pearson and Spearman correlation coefficients for each
        # proteome as a density plot
        for type,data in {"Pearson": pearson_df,
                          "Spearman": spearman_df}.items():
            ax=sns.kdeplot(
               data=data, x="correlation", hue="Type", fill=True,
               common_norm=False, alpha=0.5, linewidth=2.5
            )

            sns.move_legend(ax, "upper left")
            plt.xlabel(f"{type} correlation")
            plt.ylabel("Density")
            title = tw.fill(f"{name} - Density of {type} correlation "
                            f" coefficients for genetic code: {plot_name}", 50)
            plt.title(title)
            for ext in ["svg", "pdf"]:
                plt.savefig(f"{code_output}/{type.lower()}_plot.{ext}",
                            bbox_inches="tight")

            plt.close()
