import os
import sys
#import adjustText
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as sci
import matplotlib.pyplot as plt
import sklearn.decomposition as dec
import scipy.cluster.hierarchy as sch


if __name__=="__main__":
    path,name = sys.argv[1:4]

    codons = [f"{a}{b}{c}" for a in "ACGT" for b in "ACGT" for c in "ACGT"]
    ticks = np.array([i for i in range(0, 64, 8)] + [63])
    labels = [codons[i] for i in ticks]

    #########################################################

    vmin = 0
    vmax = 0
    context_frames = []
    for step in range(1, 10, 1):
        step_path = os.path.join(path, str(step))
        context_df = pd.read_csv(os.path.join(step_path, "codon_context.csv"),
                                 sep="\t", header=0, index_col=0)
        context_frames.append(context_df)
        vmin = min(vmin, np.min(context_df))
        vmax = max(vmax, np.max(context_df))

    fig,axes = plt.subplots(3, 3, figsize=(15, 15))
    i = 0
    j = 0
    for step,df in enumerate(context_frames):
        sns.heatmap(df, cmap="coolwarm", center=0, annot=False, cbar=True,
                    vmin=vmin, vmax=vmax, ax=axes[j,i])
        axes[j,i].set_xticks(ticks+0.5, labels, fontsize=8)
        axes[j,i].set_yticks(ticks+0.5, labels, fontsize=8)
        axes[j,i].set_title(f"Neighbor distance: {step+1}")
        i = i+1 if i < 2 else 0
        j = j+1 if i == 0 else j

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.suptitle(f"{name} - Neighboring codon counts", fontsize=14, y=0.93)

    for ext in ["svg", "pdf"]:
        plt.savefig(f"{path}/counts.{ext}", bbox_inches="tight")

    plt.close()

    #########################################################

    vmin = 0
    vmax = 0
    z_frames = []
    for df in context_frames:
        row_means = df.mean(axis=1)
        row_stds = df.std(axis=1)
        z_score_df = (df.sub(row_means, axis=0)).div(row_stds, axis=0)
        z_frames.append(z_score_df)
        vmin = min(vmin, np.min(z_score_df))
        vmax = max(vmax, np.max(z_score_df))

    fig,axes = plt.subplots(3, 3, figsize=(15, 15))
    i = 0
    j = 0
    for step,df in enumerate(z_frames):
        sns.heatmap(df, cmap="coolwarm", center=0, annot=False, cbar=True,
                    vmin=vmin, vmax=vmax, ax=axes[j,i])
        axes[j,i].set_xticks(ticks+0.5, labels, fontsize=8)
        axes[j,i].set_yticks(ticks+0.5, labels, fontsize=8)
        axes[j,i].set_title(f"Neighbor distance: {step+1}")

        df.to_csv(os.path.join(os.path.join(path, str(step+1)), "z_scores.csv"),
                  sep="\t")

        i = i+1 if i < 2 else 0
        j = j+1 if i == 0 else j

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.suptitle(f"{name} - Neighboring codon Z-scores", fontsize=14, y=0.93)

    for ext in ["svg", "pdf"]:
        plt.savefig(f"{path}/z_scores.{ext}", bbox_inches="tight")

    plt.close()
    #########################################################

    pca = dec.PCA(n_components=2)
    fig,axes = plt.subplots(3, 3, figsize=(15, 15))
    i = 0
    j = 0
    for step,df in enumerate(context_frames):
        pca_result = pca.fit_transform(df)
        axes[j,i].scatter(pca_result[:,0], pca_result[:,1], color="orange",
                          marker="x")

        '''texts = [axes[j,i].text(pca_result[k,0], pca_result[k,1],
                 codon, fontsize=6)
                 for k,codon in enumerate(df.index)]
        adjustText.adjust_text(texts, arrowprops=dict(arrowstyle="-",
                               color="gray", lw=0.5), ax=axes[j,i])'''

        axes[j,i].set_xticks([])
        axes[j,i].set_yticks([])
        axes[j,i].set_title(f"Neighbor distance: {step+1}")

        pca_df = pd.DataFrame(pca_result, columns=["PC1","PC2"], index=df.index)
        pca_df.to_csv(os.path.join(os.path.join(path, str(step+1)), "pcas.csv"),
                      sep="\t")

        i = i+1 if i < 2 else 0
        j = j+1 if i == 0 else j

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.suptitle(f"{name} - Neighboring codon PCA", fontsize=14, y=0.93)

    for ext in ["svg", "pdf"]:
        plt.savefig(f"{path}/pcas.{ext}", bbox_inches="tight")

    plt.close()

    #########################################################

    fig,axes = plt.subplots(3, 3, figsize=(15, 15))
    i = 0
    j = 0
    for step,df in enumerate(context_frames):
        linked = sch.linkage(df, method="ward")
        sch.dendrogram(linked, labels=df.index, leaf_rotation=90,
                       leaf_font_size=4, ax=axes[j,i])

        axes[j,i].set_yticks([])
        axes[j,i].set_title(f"Neighbor distance: {step+1}")
        i = i+1 if i < 2 else 0
        j = j+1 if i == 0 else j

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.suptitle(f"{name} - Neighboring codon hierarchical clustering",
                 fontsize=14, y=0.93)

    for ext in ["svg", "pdf"]:
        plt.savefig(f"{path}/clusters.{ext}", bbox_inches="tight")

    plt.close()
