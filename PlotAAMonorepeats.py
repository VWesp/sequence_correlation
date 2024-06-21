import os
import sys
import itertools
import numpy as np
import pandas as pd
import collections as col
import matplotlib.pyplot as plt

# main method
if __name__ == "__main__":
    file_path = sys.argv[1]
    output = sys.argv[2]
    plot_name= sys.argv[3]

    os.makedirs(output, exist_ok=True)

    aa_reps = pd.read_csv(file_path, sep="\t", index_col=0)
    aa_data = col.defaultdict(lambda: [[], []])
    for aa_rep in aa_reps.columns:
        if(not aa_rep in ["Length", "Genome_Tax_ID"]):
            aa = list(set(aa_rep))[0]
            aa_rep_num = np.sum(aa_reps[aa_rep])
            if(aa_rep_num > 1):
                aa_data[aa][0].append(len(aa_rep))
                aa_data[aa][1].append(aa_rep_num)

    aa_groups = {"Charged": ["R", "H", "K", "D", "E"], "Aromatic": ["F", "Y",
                 "W"], "Aliphatic": ["G", "A", "V", "L", "M", "I"],
                 "Uncharged": ["S", "T", "C", "P", "N", "Q"]}
    markers = {"Charged": "o", "Aromatic": "*", "Aliphatic": "d",
               "Uncharged": "s"}

    fig, ax = plt.subplots(figsize=(17, 10))
    x_max = 0
    y_max = 0
    x_loc = 0.76
    for group,amino_acids in aa_groups.items():
        color_space = plt.cm.rainbow(np.linspace(0, 1, len(amino_acids)))
        color_map = {amino_acids[i]:c for i, c in enumerate(color_space)}
        handles = []
        for aa in amino_acids:
            aa_rep_lengths = np.asarray(aa_data[aa][0])
            aa_rep_numbers = np.asarray(aa_data[aa][1])
            log_reps = np.asarray(np.log2(aa_rep_numbers))
            handle = ax.scatter(aa_rep_lengths, log_reps, s=80,
                                color=color_map[aa], alpha=0.7, label=aa,
                                marker=markers[group], edgecolor="black")
            handles.append(handle)

            x_max = max(x_max, max(aa_rep_lengths))
            y_max = max(y_max, max(log_reps))

        legend = ax.legend(handles=handles, title=group+"\namino acids",
                           bbox_to_anchor=(x_loc, 1))
        ax.add_artist(legend)
        x_loc += 0.08

    ax.set_xlim(0, x_max*1.05)
    ax.set_ylim(0, y_max*1.05)
    ax.grid(visible=True, which="major", color="#999999",
             linestyle="dotted", alpha=0.5, zorder=0)
    ax.set_xlabel("Repeat length")
    ax.set_ylabel("log2-Number of monorepeats")
    ax.set_title(plot_name+", Number of monorepeats for given amino acid")
    # output
    plt.savefig(os.path.join(output, "monorepeats.svg"), bbox_inches="tight",
                dpi=fig.dpi)
    plt.savefig(os.path.join(output, "monorepeats.pdf"), bbox_inches="tight",
                dpi=fig.dpi)
    plt.close()
