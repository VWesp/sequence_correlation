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

    for aa in amino_acids:
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
        plt.close()
