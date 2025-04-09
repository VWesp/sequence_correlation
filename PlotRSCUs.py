import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__=="__main__":
    path_to_data = sys.argv[1]

    kingdoms = ["Archaea", "Bacteria", "Eukaryotes", "Viruses"]
    fig,axes = plt.subplots(2, 2)
    i = 0
    j = 0
    for king in kingdoms:
        king_path = os.path.join(path_to_data, king)
        data = pd.read_csv(os.path.join(king_path, "mean_RSCUs.csv"), sep="\t",
                           header=0, index_col=0).fillna(0)
        sns.heatmap(data, cmap="viridis", ax=axes[i,j])
        axes[i,j].set_xticks(np.arange(len(data.columns))+0.5, data.columns,
                             fontsize=10, va="center")
        axes[i,j].set_yticks(np.arange(len(data.index))+0.5, data.index,
                             fontsize=4, va="center")
        axes[i,j].set_title(king, fontweight="bold", fontsize=14)
        i = 1 if j == 1 else i
        j = 0 if j == 1 else j + 1

    fig.subplots_adjust(wspace=0.1)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    fig.suptitle("Mean relative synonymous codon usage (RSCU) across kingdoms",
                 y=0.96, fontsize=18)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(path_to_data, f"rscus.{ext}"),
                    bbox_inches="tight")
