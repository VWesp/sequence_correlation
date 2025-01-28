import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import collections as col
import matplotlib.pyplot as plt


if __name__ == "__main__":
    path,name =sys.argv[1:3]
    output = os.path.dirname(path)
    amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
                   "A", "G", "P", "T", "V", "L", "R", "S"]

    data = None
    with open(path, "r", encoding="utf-8") as reader:
        data = json.load(reader)

    ordered_data = col.defaultdict(lambda: col.defaultdict(int))
    for _,aa_data in data.items():
        for aa in amino_acids:
            if(aa in aa_data):
                for rep_len,num in aa_data[aa].items():
                    ordered_data[aa][int(rep_len)] += num

    rep_df = pd.DataFrame(ordered_data).fillna(0).sort_index()
    rep_df = rep_df.drop([1])

    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(np.log10(rep_df+1), cmap="viridis", rasterized=True)
    plt.xlabel("Amino acid")
    plt.ylabel("Repeat length")
    plt.yticks(rotation=0)
    plt.title(f"{name} - Heatmap of log10-sum proteomic amino acid repeat count")
    for ext in ["svg", "pdf"]:
        plt.savefig(f"{os.path.join(output, f"sum_repeats")}.{ext}",
                    bbox_inches="tight")

    plt.close()


    '''fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("rainbow")
    for i, (aa,rep_data) in enumerate(ordered_data.items()):
        for rep,nums in rep_data.items():
            ax.scatter(
                [i+1] * len(nums),
                [rep] * len(nums),
                nums,
                label=aa if rep==1 else "",
                color=cmap(i/len(amino_acids))
            )

    ax.set_xticks(range(len(amino_acids)))
    ax.set_xticklabels(amino_acids)
    ax.set_xlabel("Amino acid")
    ax.set_ylabel("Repeat length")
    ax.set_zlabel("Count")
    ax.set_title(f"{name} - Scatter plot of mean proteomic amino repeats")
    plt.legend(loc="upper center", fontsize=8, ncol=4, fancybox=True,
               shadow=True)

    for ext in ["svg", "pdf"]:
        plt.savefig(f"test.{ext}", bbox_inches="tight")

    plt.close()'''
