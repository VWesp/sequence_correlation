import os
import sys
import numpy as np
import sympy as sp
import pandas as pd
import seaborn as sns
import collections as col
import matplotlib.pyplot as plt
from Bio.Data import CodonTable


if __name__ == "__main__":
    path_to_data = sys.argv[1]
    genetic_id = int(sys.argv[2])
    output = sys.argv[3]
    plot_name = sys.argv[4]

    #os.makedirs(output, exist_ok=True)

    aa_dis_df = pd.read_csv(path_to_data, sep="\t", index_col=0, dtype=str)
    aa_dis_df["GC"] = aa_dis_df["GC"].astype(float)
    aa_dis_df = aa_dis_df.sort_values("GC")

    # list of amino acids
    amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
                   "A", "G", "P", "T", "V", "L", "R", "S"]

    aa_gc_data_freq = col.defaultdict(lambda: col.defaultdict(lambda: 0))
    gt_ids = set(aa_dis_df["Genome_Tax_ID"])
    for gt_id in gt_ids:
        gt_id_data = aa_dis_df[aa_dis_df["Genome_Tax_ID"]==gt_id]
        gt_id_aa_ave = col.defaultdict(lambda: 0)
        for aa in amino_acids:
            gt_id_aa_num = []
            for codon_num in gt_id_data[aa]:
                prot_aa_num = sum([int(index.split(":")[1])
                                   for index in codon_num.split(";")])
                gt_id_aa_num.append(prot_aa_num)

            gt_id_aa_ave[aa] = np.sum(gt_id_aa_num)

        for aa in amino_acids:
            gc_ave = np.average(gt_id_data["GC"])
            ave_aa_freq = gt_id_aa_ave[aa] / np.sum(list(gt_id_aa_ave.values()))
            aa_gc_data_freq[aa][gc_ave] = ave_aa_freq

    sorted_gc_values = np.asarray(sorted(list(aa_gc_data_freq["M"].keys())))

    # get the genetic code given the ID
    codon_table = CodonTable.unambiguous_dna_by_id[genetic_id]
    genetic_name = " ".join(codon_table.names[:-1])
    g = sp.symbols("g", float=True)
    aa_gc_formulas = col.defaultdict(lambda: 0)
    nuc_formulas = {"A": (1-g)/2, "C": g/2, "G": g/2, "T": (1-g)/2}
    for codon,aa in codon_table.forward_table.items():
        codon_formula = 1
        for nuc in codon:
            codon_formula *= nuc_formulas[nuc]

        aa_gc_formulas[aa] += codon_formula

    aa_gc_calc_freq = {}
    for aa,formula in aa_gc_formulas.items():
        # create the frequency equation from the function
        aa_equation = sp.Eq(g, formula)
        # turn the frequency equation into a numpy function
        aa_np_equation = sp.lambdify(g, aa_equation.rhs, "numpy")
        # calculate the frequency for all GC contents
        aa_freq_values = aa_np_equation(sorted_gc_values)
        aa_gc_calc_freq[aa] = aa_freq_values

    '''for aa in amino_acids:
        freq_data = aa_gc_data_freq[aa]
        sorted_freq_data = dict(sorted(freq_data.items()))
        gcs = np.asarray(list(sorted_freq_data.keys()))
        data_aa_freqs = np.asarray(list(sorted_freq_data.values()))
        data_aa_freqs_fit = np.poly1d(np.polyfit(gcs, data_aa_freqs, 3))(gcs)
        calc_aa_freqs = aa_gc_calc_freq[aa]
        plt.scatter(gcs, data_aa_freqs, color="red", alpha=0.5,
                    edgecolor="black", label="Genome data")

        plt.plot(gcs, data_aa_freqs_fit, color="black", linewidth=3)
        plt.plot(gcs, data_aa_freqs_fit, color="#44a5c2", linewidth=2,
                 label="Fitted genome data")

        calc_aa_freqs = aa_gc_calc_freq[aa]
        plt.plot(gcs, calc_aa_freqs, color="black", linewidth=3)
        plt.plot(gcs, calc_aa_freqs, color="#ffae49", linewidth=2,
                 label="Genetic data")

        # x label
        plt.xlabel("GC content")
        # y label
        plt.ylabel("Amino acid frequency")
        title = "{}, Amino acid: {}, Code: {}".format(plot_name, aa,
                                                      genetic_name)
        plt.title(title)
        plt.legend(loc="best")

        # output
        plot_output = os.path.join(output,
                               aa+"_"+"_".join(genetic_name.lower().split(" ")))
        plt.savefig(plot_output+".svg", bbox_inches="tight")
        plt.savefig(plot_output+".pdf", bbox_inches="tight")
        plt.close()'''

    fitted_data = pd.DataFrame({"GC": [], "AminoAcid": [],
                                "Type": [], "Frequency": []})
    theoretical_data = pd.DataFrame({"GC": [], "AminoAcid": [],
                                     "Type": [], "Frequency": []})
    for aa in amino_acids:
        freq_data = aa_gc_data_freq[aa]
        sorted_freq_data = dict(sorted(freq_data.items()))
        gcs = np.asarray(list(sorted_freq_data.keys()))
        data_aa_freqs = np.asarray(list(sorted_freq_data.values()))
        data_aa_freqs_fit = np.poly1d(np.polyfit(gcs, data_aa_freqs, 3))(gcs)
        calc_aa_freqs = aa_gc_calc_freq[aa]
        amino = [aa] * len(gcs)

        new_rows = pd.DataFrame({"GC": gcs, "AminoAcid": amino,
                              "Type": "Fitted", "Frequency": data_aa_freqs_fit})
        fitted_data = pd.concat([fitted_data, new_rows], ignore_index=True)

        new_rows = pd.DataFrame({"GC": gcs, "AminoAcid": amino,
                              "Type": "Calculated", "Frequency": calc_aa_freqs})
        theoretical_data = pd.concat([theoretical_data, new_rows],
                                      ignore_index=True)

    combined_data = pd.concat([fitted_data, theoretical_data])
    heatmap_data = combined_data.pivot_table(index=["AminoAcid", "Type"],
                               columns="GC", values="Frequency", aggfunc="mean")

    plt.figure(figsize=(14, 16))
    ax = sns.heatmap(heatmap_data, cmap="YlGnBu",
                     cbar_kws={"label": "Frequency"})
    new_labels = [label[0] if i % 2 == 0 else ""
                  for i,label in enumerate(heatmap_data.index)]
    #ax.set_yticklabels(new_labels)
    ax.hlines(np.arange(0, 41, 2), *ax.get_xlim(), colors="black")
    plt.title("Heatmap of New Theoretical and Fitted Amino Acid Frequency vs. GC Content")
    plt.xlabel("GC Content")
    plt.ylabel("Amino Acid and Type")
    plt.show()
