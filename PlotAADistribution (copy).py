import os
import sys
import math
import numpy as np
import sympy as sp
import pandas as pd
import seaborn as sns
import textwrap as tw
import collections as col
import scipy.stats as sci
import equation_functions as ef
from Bio.Data import CodonTable
import matplotlib.pyplot as plt
import matplotlib.patches as patch

# Function to plot and save heatmaps
def plot_heatmap(df, types, cmap, label, xticklabels, yticks, yticklabels,
                 title, name, code_output):
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        df[types], cmap=cmap, cbar_kws={"label": label}, ax=ax
    )

    # Add vertical lines between amino acid columns
    ax.vlines(np.arange(1, len(types)), *ax.get_ylim(),
              colors="white", lw=1)

    # Set x-ticks, y-ticks, and labels
    ax.set_xticks(np.arange(0.5, len(xticklabels), 1))
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, rotation=0)
    ax.set_xlabel("Amino acid")
    ax.set_ylabel("GC content")

    title = title = tw.fill(title, 50)
    ax.set_title(title)

    # Save the heatmap in both SVG and PDF formats
    for ext in ["svg", "pdf"]:
        plt.savefig(f"{code_output}/{name}.{ext}", bbox_inches="tight")
    plt.close()

# Main function for analysis
def main(path_to_file, output_dir, plot_name):
    os.makedirs(output_dir, exist_ok=True)
    aa_dis_df = pd.read_csv(path_to_file, sep="\t", index_col=0, dtype=str)

    # Initialize list of amino acids and iterate over genetic codes
    amino_acids = None
    for code_id, codon_table in CodonTable.unambiguous_dna_by_id.items():
        genetic_name = " ".join(codon_table.names[:-1]).lower().replace(
            " ", "_")
        code_output = os.path.join(output_dir, genetic_name)
        os.makedirs(code_output, exist_ok=True)

        # Map amino acids to their corresponding codons
        aa_to_codon = col.defaultdict(list)
        for codon, aa in codon_table.forward_table.items():
            aa_to_codon[aa].append(codon)

        # Count the number of codons for each amino acid
        codon_count = col.Counter(codon_table.forward_table.values())

        # Initialize and sort amino acids list based on the first genetic code
        if amino_acids is None:
            amino_acids = sorted(codon_count, key=lambda x: (codon_count[x],
                                                             x))

        # Calculate the frequency percentage for each amino acid based on the
        # codon number
        total_codons = sum(codon_count.values())
        genetic_code_num = np.array([codon_count[aa] / total_codons
                                     for aa in amino_acids])

        # Calculate the frequency percentage for each amino acid based on the
        # metabolic cost
        costs = {"A": 11.7, "C": 24.7, "D": 12.7, "E": 15.3, "F": 52.0,
                 "G": 11.7, "H": 38.3, "I": 32.3, "K": 30.3, "L": 27.3,
                 "M": 34.3, "N": 14.7, "P": 20.3, "Q": 16.3, "R": 27.3,
                 "S": 11.7, "T": 18.7, "V": 23.3, "W": 74.3, "Y": 50.0}
        costs_inv = {aa:1/costs[aa] for aa in amino_acids}
        costs_inv_norm = [costs_inv[aa]/sum(list(costs_inv.values()))
                          for aa in amino_acids]

        # Initialize the plot for amino acid distribution
        fig, ax = plt.subplots()
        max_codon_num = max(codon_count.values())
        color_map = plt.cm.rainbow(np.linspace(0, 1, max_codon_num))

        # Create legend patches
        patches = []
        for color in color_map:
            patches.append(patch.Patch(facecolor=color, edgecolor="black",
                           hatch="//"))
            patches.append(patch.Patch(facecolor="white", edgecolor="black",
                           hatch="\\\\"))

        patches += [
            patch.Patch(facecolor="grey", edgecolor="black", hatch="//"),
            patch.Patch(facecolor="white", edgecolor="black", hatch="\\\\")
        ]

        # Accumulate amino acid counts
        sum_aa = col.defaultdict(int)
        for l_index, aa in enumerate(amino_acids):
            codon_dis = col.defaultdict(int)

            # Count occurrences of each codon for the current amino acid
            for codon_num in aa_dis_df[aa]:
                for codon_entry in codon_num.split(";"):
                    codon, count = codon_entry.split(":")
                    if codon != "XXX":
                        codon_dis[codon] += int(count)

            # Plot observed counts with color for canonical codons
            bottom = 0
            for c_index, codon in enumerate(aa_to_codon[aa]):
                cod_num = codon_dis[codon]
                sum_aa[aa] += cod_num
                ax.bar(l_index - 0.15, cod_num, width=0.3, edgecolor="black",
                       linewidth=0.75, bottom=bottom, color=color_map[c_index],
                       hatch="//")
                bottom += cod_num

            # Plot non-canonical codons in grey
            for codon, num in codon_dis.items():
                if codon not in aa_to_codon[aa]:
                    sum_aa[aa] += num
                    ax.bar(l_index - 0.15, num, width=0.3, edgecolor="black",
                           linewidth=0.75, bottom=bottom, color="grey",
                           hatch="//")
                    bottom += num

        # Plot the expected amino acid counts
        sum_aa_list = list(sum_aa.values())
        x_data = np.arange(len(amino_acids))
        genetic_calc_list = genetic_code_num * sum(sum_aa_list)
        ax.bar(x_data + 0.15, genetic_calc_list, width=0.3, color="white",
               edgecolor="black", linewidth=0.75, hatch="\\\\")

        # Calculate and plot differences between observed and expected counts
        dif_text_max_high = max(max(sum_aa_list), max(genetic_calc_list))
        dif_text_upper = dif_text_max_high * 0.01
        dif_text_lower = dif_text_max_high * 0.02
        dif_text_num = dif_text_max_high * 0.03

        for index in x_data:
            dif_text_high = max(sum_aa_list[index], genetic_calc_list[index])
            left_text = x_data[index] - 0.15
            right_text = x_data[index] + 0.15
            bar_x = [left_text - 0.15, left_text - 0.15,
                     right_text + 0.15, right_text + 0.15]
            bar_y = [sum_aa_list[index] + dif_text_upper,
                     dif_text_high + dif_text_lower,
                     dif_text_high + dif_text_lower,
                     genetic_calc_list[index] + dif_text_upper]
            plt.plot(bar_x, bar_y, "k-", lw=1)

            # Calculate percentage difference between observed and expected
            data_dif_perc = (sum_aa_list[index] - genetic_calc_list[index]) / (
                (sum_aa_list[index] + genetic_calc_list[index]) / 2) * 100

            plt.text((left_text + right_text) / 2,
                     dif_text_high + dif_text_num, f"{data_dif_perc:.1f}",
                     size=4, ha="center", va="bottom")

        # Calculate Pearson and Spearman correlation coefficients
        pcc, pcc_p = sci.pearsonr(sum_aa_list, genetic_code_num)
        r2, r2_p = sci.spearmanr(sum_aa_list, genetic_code_num)

        # Add correlation text box to the plot
        ax.text(1.02, 0.98,
                f"pcc: {pcc:.2f}; $p_{{pcc}}$: {pcc_p:.1e}\n"
                f"$r2$: {r2:.2f}; $p_{{r2}}$: {r2_p:.1e}",
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white",
                          edgecolor="grey", alpha=0.5))

        # Set up legend, labels, and title
        plt.legend(handles=patches,
                   labels=[""] * (max_codon_num * 2) +
                   ["Genomic data", "Genetic code"],
                   ncol=max_codon_num + 1, handletextpad=0.5,
                   handlelength=1.0, columnspacing=-0.5, loc="upper left")
        plt.xlabel("Amino acid")
        plt.ylabel("Summed amino acid number")
        plt.xticks(np.arange(len(amino_acids)), amino_acids)

        genetic_name = genetic_name.capitalize().replace("_", " ")
        title = tw.fill(f"{plot_name} - Correlation between amino acid "
                        f"and codon number for genetic code: {genetic_name}",
                        50)
        plt.title(title)

        for ext in ["svg", "pdf"]:
            plt.savefig(f"{code_output}/barplot.{ext}", bbox_inches="tight")
        plt.close()

        # Prepare symbols and functions for expected frequency calculations
        g = sp.symbols("g", float=True)
        funcs = ef.build_functions(aa_to_codon)["amino"]

        # Iterate over each proteome to calculate amino acid counts
        aa_count_lst = []
        for proteome_id in set(aa_dis_df["Genome_Tax_ID"]):
            prot_aa_count = aa_dis_df[aa_dis_df["Genome_Tax_ID"] ==
                                      proteome_id]
            aa_count_dic = {}

            # Calculate the observed amino acid counts
            for aa in amino_acids:
                aa_count_dic[aa] = np.median([
                    sum(int(c.split(":")[1]) for c in codon_num.split(";")
                        if c.split(":")[0] != "XXX")
                    for codon_num in prot_aa_count[aa]
                ])

            # Calculate the overall amino acid sum and average GC content
            aa_sum = sum(aa_count_dic.values())
            aa_count_dic["GC_ave"] = np.mean(prot_aa_count["GC"].astype(float))

            # Calculate expected counts, residuals, and log-transformed values
            for aa in amino_acids:
                aa_exp = aa + "_expected"
                aa_res = aa + "_residual"
                aa_count_dic[aa] = np.log2(aa_count_dic[aa] + 1)
                expected_count = float(funcs[aa].subs(g,
                                                      aa_count_dic["GC_ave"])
                                       ) * aa_sum
                aa_count_dic[aa_exp] = np.log2(expected_count)
                aa_count_dic[aa_res] = (aa_count_dic[aa] -
                                        aa_count_dic[aa_exp])

            # Calculate Pearson and Spearman correlations with p-values
            obs_aa_count = [aa_count_dic[aa] for aa in amino_acids]
            exp_aa_count = [aa_count_dic[aa + "_expected"]
                            for aa in amino_acids]
            aa_count_dic["Pearson"], aa_count_dic["p-value: Pearson"] = (
                sci.pearsonr(obs_aa_count, exp_aa_count))
            aa_count_dic["Spearman"], aa_count_dic["p-value: Spearman"] = (
                sci.spearmanr(obs_aa_count, exp_aa_count))

            aa_count_lst.append(aa_count_dic)

        # Create DataFrame from the list of dictionaries and sort by "GC_ave"
        exp_amino_acids = [aa + "_expected" for aa in amino_acids]
        res_amino_acids = [aa + "_residual" for aa in amino_acids]
        columns = (["GC_ave"] + amino_acids + exp_amino_acids +
                   res_amino_acids + ["Pearson", "p-value: Pearson",
                                      "Spearman", "p-value: Spearman"])

        aa_proteome_df = pd.DataFrame(aa_count_lst, columns=columns)
        aa_proteome_df = aa_proteome_df.sort_values("GC_ave", ascending=False)

        # Define y-ticks and y-tick labels based on GC content
        yticks = np.linspace(0, len(aa_proteome_df["GC_ave"]) - 1, 20,
                             dtype=int)
        yticklabels = [f"{aa_proteome_df['GC_ave'].iloc[idx]:.2f}"
                       for idx in yticks]

        # Plot observed, expected, and residual heatmaps
        plot_heatmap(aa_proteome_df, amino_acids, "YlGnBu",
                     "log2-Median amino acid count", amino_acids, yticks,
                     yticklabels, f"{plot_name} - Observed amino acid count "
                     f"for genetic code: {genetic_name}", "obs_heatmap",
                     code_output)

        plot_heatmap(aa_proteome_df, exp_amino_acids, "YlGnBu",
                     "log2-Median amino acid count", amino_acids, yticks,
                     yticklabels, f"{plot_name} - Expected amino acid count "
                     f"for genetic code: {genetic_name}", "exp_heatmap",
                     code_output)

        plot_heatmap(aa_proteome_df, res_amino_acids, "coolwarm",
                     "Residual (Observed - Expected)", amino_acids, yticks,
                     yticklabels, f"{plot_name} - Residual between observed "
                     f"and expected count for genetic code: {genetic_name}",
                     "res_heatmap", code_output)

        # Plotting Pearson and Spearman correlation coefficients vs. GC content
        fig, ax = plt.subplots(figsize=(10, 6))
        pearson_df = aa_proteome_df[aa_proteome_df["p-value: Pearson"] < 0.05]
        spearman_df = aa_proteome_df[aa_proteome_df["p-value: Spearman"] <
                                     0.05]

        ax.scatter(pearson_df["GC_ave"], pearson_df["Pearson"], alpha=0.7,
                   color="red", edgecolor="black", label="Pearson correlation")

        ax.scatter(spearman_df["GC_ave"], spearman_df["Spearman"], alpha=0.7,
                   color="blue", edgecolor="black", label="Spearman correlation")

        # Set plot labels and title
        cor_title = tw.fill(f"{plot_name} - Pearson/Spearman correlation "
                            f"for genetic code: {genetic_name}", 50)
        ax.set_xlabel("GC content")
        ax.set_ylabel("Pearson/Spearman correlation coefficient")
        ax.set_title(cor_title)
        plt.legend(loc="upper left")

        # Save the correlation plot
        for ext in ["svg", "pdf"]:
            plt.savefig(f"{code_output}/correlation.{ext}",
                        bbox_inches="tight")
        plt.close()

        # Save the DataFrame to a CSV file
        aa_proteome_df.to_csv(f"{code_output}/data_heatmap.csv", sep="\t")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
