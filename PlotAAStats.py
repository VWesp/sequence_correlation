import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as sci
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.patheffects as path_effects


def optimal_bin(data):
    data = data.to_numpy()
    iqr = np.quantile(data, 0.75) - np.quantile(data, 0.25)
    h = 2 * iqr / len(data)**(1/3)
    return int((data.max() - data.min()) / h) + 1


def mean_corr(corrs, p_values):
    # Fisher's Z-transformation
    z_values = [0.5*np.log((1+r)/(1-r)) for r in corrs]
    mean_z = np.mean(z_values)
    mean_corr = (np.exp(2*mean_z)-1) / (np.exp(2*mean_z)+1)
    return [mean_corr, np.mean(p_values)]


def plot_lengths(data, domain, output):
    lengths = np.log10(data["Length_mean"]+1)
    bins = optimal_bin(lengths)
    sns.histplot(lengths, bins=bins, alpha=0.4, color="maroon", kde=True,
                 line_kws={"linewidth": 2, "linestyle": "--"})
    plt.title(f"{domain} - Distribution of mean protein length")
    plt.xlabel("log10(Length)", fontweight="bold", fontsize=10)
    plt.ylabel("Count", fontweight="bold", fontsize=10)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(output, f"protein_lengths.{ext}"),
                    bbox_inches="tight")

    plt.close()


def plot_gcs(data, domain, output):
    gcs = data["GC_mean"]
    bins = optimal_bin(gcs)
    sns.histplot(gcs, bins=bins, alpha=0.4, color="maroon", kde=True,
                 line_kws={"linewidth": 2, "linestyle": "--"})
    plt.title(f"{domain} - Distribution of mean protein GC content")
    plt.xlabel("GC content", fontweight="bold", fontsize=10)
    plt.ylabel("Count", fontweight="bold", fontsize=10)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(output, f"protein_gcs.{ext}"),
                    bbox_inches="tight")

    plt.close()


def plot_amount(data, domain, output):
    nums = np.log10(data["#Proteins"]+1)
    bins = optimal_bin(nums)
    sns.histplot(nums, bins=bins, alpha=0.4, color="maroon", kde=True,
                 line_kws={"linewidth": 2, "linestyle": "--"})
    plt.title(f"{domain} - Distribution of protein amount")
    plt.xlabel("log10(Amount)", fontweight="bold", fontsize=10)
    plt.ylabel("Count", fontweight="bold", fontsize=10)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(output, f"protein_amount.{ext}"),
                    bbox_inches="tight")

    plt.close()


def plot_pct(data, aa_groups, domain, output):
    fig,axes = plt.subplots(2, 2, sharey=True)
    i = 0
    j = 0
    for aa_type,aa_list in aa_groups.items():
        aa_pct_code_cols = [f"{aa}_pct_code" for aa in aa_list]
        x_pos = np.arange(len(aa_list)) - 0.17
        c = "royalblue"
        code_box = axes[i,j].boxplot(data[aa_pct_code_cols], positions=x_pos,
                                     widths=0.3, notch=True, patch_artist=True,
                                     boxprops=dict(facecolor=c, color="black"),
                                     capprops=dict(color=c), whiskerprops=dict(color=c),
                                     flierprops=dict(color=c, markeredgecolor=c),
                                     medianprops=dict(color=c), zorder=2)

        aa_pct_freq_cols = [f"{aa}_pct_freq" for aa in aa_list]
        x_pos = np.arange(len(aa_list)) + 0.18
        c = "goldenrod"
        freq_box = axes[i,j].boxplot(data[aa_pct_freq_cols], positions=x_pos,
                                     widths=0.3,  notch=True, patch_artist=True,
                                     boxprops=dict(facecolor=c, color="black"),
                                     capprops=dict(color=c), whiskerprops=dict(color=c),
                                     flierprops=dict(color=c, markeredgecolor=c),
                                     medianprops=dict(color=c), zorder=2)

        axes[i,j].axhline(y=0, zorder=1, color="firebrick", linestyle="--")
        axes[i,j].grid(visible=True, which="major", color="#999999",
                       linestyle="dotted", alpha=0.5, zorder=0)
        axes[i,j].set_xticks(np.arange(len(aa_list)), aa_list)
        axes[i,j].set_xlabel("Amino acid")
        axes[i,j].set_title(f"{aa_type} amino acids")

        if(j == 0):
            axes[i,j].set_ylabel("Percentage difference in %")
            if(i == 0):
                axes[i,j].legend([code_box["boxes"][0], freq_box["boxes"][0]],
                                 ["Codon number", "Codon+GC"],
                                 bbox_to_anchor=(1.45, 1.02), fancybox=True,
                                 fontsize=12)

        axes[i,j].yaxis.set_tick_params(labelleft=True)
        j = 1 if i == 1 else j
        i = 0 if i == 1 else i + 1

    spear_corr,spear_corr_p = mean_corr(data["Spearman_code"],
                                        data["Spearman_code_p"])
    kendall_corr,kendall_corr_p = mean_corr(data["Kendall_code"],
                                            data["Kendall_code_p"])
    axes[0,0].text(1.03, 0.6, f"Codon number\n"
        f"  - Spearman:\n"
        f"    - Coefficient: {spear_corr:.5f}\n"
        f"    - p-value: {spear_corr_p:.3e}\n"
        f"\n  - Kendall's Tau:\n"
        f"    - Coefficient: {kendall_corr:.5f}\n"
        f"    - p-value: {kendall_corr_p:.3e}",
        transform=axes[0,0].transAxes, fontsize=11,
        verticalalignment="top", linespacing=1.5,
        bbox=dict(boxstyle="round", facecolor="white",
                  edgecolor="grey", alpha=0.5))

    spear_corr,spear_corr_p = mean_corr(data["Spearman_freq"],
                                        data["Spearman_freq_p"])
    kendall_corr,kendall_corr_p = mean_corr(data["Kendall_freq"],
                                            data["Kendall_freq_p"])
    axes[0,0].text(1.03, 0.0, f"Codon+GC\n"
        f"  - Spearman:\n"
        f"    - Coefficient: {spear_corr:.5f}\n"
        f"    - p-value: {spear_corr_p:.3e}\n"
        f"\n  - Kendall's Tau:\n"
        f"    - Coefficient: {kendall_corr:.5f}\n"
        f"    - p-value: {kendall_corr_p:.3e}",
        transform=axes[0,0].transAxes, fontsize=11,
        verticalalignment="top", linespacing=1.5,
        bbox=dict(boxstyle="round", facecolor="white",
                  edgecolor="grey", alpha=0.5))

    fig.subplots_adjust(wspace=0.6, hspace=0.3)
    title = f"{domain} - Percentage difference between amino acid distributions"
    fig.suptitle(title, fontsize=15, y=0.95)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(output, f"pct_difference.{ext}"),
                    bbox_inches="tight")

    plt.close()


def plot_corr_coefficients(data, output):
    color_palette = {"CN-spear": "maroon", "GC-spear": "darkorange",
                     "CN-kendall": "royalblue", "GC-kendall": "forestgreen"}
    data["Comb_col"] = data["Correlation"] + "-" + data["Comparison"]
    g = sns.FacetGrid(data, row="Domain", hue="Comb_col",
                      palette=list(color_palette.values()))
    g.map(sns.kdeplot, "Coefficient", clip_on=False, fill=False, alpha=1,
          color="black")
    g.map(sns.kdeplot, "Coefficient", clip_on=False, fill=True, alpha=0.5,
          hatch="x")
    g.refline(y=0, linewidth=2, linestyle="-", color="grey", clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, x.iloc[0], color="black", fontsize=14, ha="left",
                va="center", fontweight="bold", transform=ax.transAxes)

    g.map(label, "Domain")
    g.set_titles("")
    g.set_xlabels(label="Correlation coefficient", fontweight="bold",
                  fontsize=14)
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    legend_patches = [
        patch.Patch(color=color_palette["CN-spear"],
                    label="Spearman: Codon number"),
        patch.Patch(color=color_palette["GC-spear"],
                    label="Spearman: Codon+GC"),
        patch.Patch(color=color_palette["CN-kendall"],
                    label="Kendall's Tau: Codon number"),
        patch.Patch(color=color_palette["GC-kendall"],
                    label="Kendall's Tau: Codon+GC")
    ]
    plt.legend(handles=legend_patches, bbox_to_anchor=(0.75, 4.88),
               fancybox=True, fontsize=12, ncols=2)

    g.fig.set_figheight(10)
    g.fig.set_figwidth(15)

    title = "Correlation coefficient distributions across domains"
    plt.suptitle(title, x=0.55, y=1.08, fontsize=18)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(output, f"corr_coefficients.{ext}"),
                    bbox_inches="tight")

    plt.close()


def plot_plotogram(data_dct, aa_groups, output):
    amino_acids = [aa for group,aa_list in aa_groups.items()
                   for aa in aa_list]
    aa_mean_cols = [f"{aa}_mean" for aa in amino_acids]
    aa_pct_code_cols = [f"{aa}_pct_code" for aa in amino_acids]
    aa_pct_freq_cols = [f"{aa}_pct_freq" for aa in amino_acids]
    aa_group_order = [aa for group,aa_list in aa_groups.items()
                      for aa in aa_list]
    fig,axes = plt.subplots(5, 4)
    sns_pal = sns.color_palette("viridis", len(amino_acids))
    for j,(domain,data) in enumerate(data_dct.items()):
        #Plot protein GC content
        gcs = data["GC_mean"]
        bins = optimal_bin(gcs)
        sns.histplot(gcs.values, bins=bins, alpha=0.4, color="maroon",
                     line_kws={"linewidth": 2, "linestyle": "--"}, zorder=3,
                     ax=axes[0,j])
        axes[0,j].set_xlabel("GC content", fontweight="bold")

        # Plot amino acid distributions
        dis_data = data[aa_mean_cols].melt(var_name="x", value_name="y")
        sns.boxplot(data=dis_data, x="x", y="y", hue="x", showfliers=False,
                    palette=sns_pal, zorder=2, ax=axes[1,j])
        axes[1,j].set_xticks(np.arange(len(amino_acids)), amino_acids,
                             fontsize=8)
        axes[1,j].set_xlabel("Amino acid", fontweight="bold")
        group_pos = 0
        last_pos = 0
        for index,(group,aa_list) in enumerate(aa_groups.items()):
            group_pos += len(aa_list)
            if(group_pos < 20):
                axes[1,j].axvline(x=group_pos-0.5, zorder=1, color="firebrick",
                                  linestyle="--", linewidth=1)

            text_pos = (group_pos+last_pos) / 20 / 2
            text = axes[1,j].text(text_pos-0.02, 0.95, index+1, color="firebrick",
                                  fontweight="bold", transform=axes[1,j].transAxes)
            text.set_path_effects([
                path_effects.Stroke(linewidth=1, foreground="white"),
                path_effects.Normal()
            ])
            last_pos = group_pos

        # Plot amino acid percentage differences
        pct_code_data = data[aa_pct_code_cols].melt(var_name="x", value_name="y")
        pct_code_data["z"] = ["Codon number"] * len(pct_code_data)
        pct_code_data["x"] = pct_code_data["x"].str.split("_").str[0]
        pct_freq_data = data[aa_pct_freq_cols].melt(var_name="x", value_name="y")
        pct_freq_data["z"] = ["Codon+GC"] * len(pct_freq_data)
        pct_freq_data["x"] = pct_freq_data["x"].str.split("_").str[0]
        pct_data = pd.concat([pct_code_data, pct_freq_data])
        sns.lineplot(data=pct_data, x="x", y="y", hue="z", style="z",
                     errorbar="pi", markers=True, dashes=False, alpha=0.8,
                     palette=["royalblue", "goldenrod"], legend=None,
                     zorder=2, ax=axes[2,j])
        axes[2,j].axhline(y=0, zorder=1, color="firebrick", linestyle="--")
        axes[2,j].set_xticks(np.arange(len(amino_acids)), amino_acids,
                             fontsize=8)
        axes[2,j].set_xlabel("Amino acid", fontweight="bold")
        group_pos = 0
        last_pos = 0
        for index,(group,aa_list) in enumerate(aa_groups.items()):
            group_pos += len(aa_list)
            if(group_pos < 20):
                axes[2,j].axvline(x=group_pos-0.5, zorder=1, color="firebrick",
                                  linestyle="--", linewidth=1)

            text_pos = (group_pos+last_pos) / 20 / 2
            text = axes[2,j].text(text_pos-0.02, 0.95, index+1, color="firebrick",
                                  fontweight="bold", transform=axes[2,j].transAxes)
            text.set_path_effects([
                path_effects.Stroke(linewidth=1, foreground="white"),
                path_effects.Normal()
            ])
            last_pos = group_pos

        # Plot Spearman correlation coefficients
        spear_corr = data[["Spearman_code", "Spearman_freq"]].melt(var_name="x",
                                                                   value_name="y")
        sns.violinplot(data=spear_corr, x="y", y="x", hue="x", split=True,
                       orient="h", palette=["royalblue", "goldenrod"],
                       legend=None, zorder=2, ax=axes[3,j])
        axes[3,j].set_xlabel("Correlation coefficient ($r_S$)", fontweight="bold")
        axes[3,j].tick_params(axis="y", labelleft=False)

        # Plot Kendall's Tau correlation coefficients
        kendall_corr = data[["Kendall_code", "Kendall_freq"]].melt(var_name="x",
                                                                   value_name="y")
        sns.violinplot(data=kendall_corr, x="y", y="x", hue="x", split=True,
                       orient="h", palette=["royalblue", "goldenrod"],
                       legend=None, zorder=2, ax=axes[4,j])
        axes[4,j].set_xlabel(u"Correlation coefficient (\u03C4)", fontweight="bold")
        axes[4,j].tick_params(axis="y", labelleft=False)

        axes[0,j].set_title(domain, fontsize=16, fontweight="bold")
        for i in range(5):
            axes[i,j].set_ylabel("")

    # Set the y-label for each row
    axes[0,0].set_ylabel("Count", fontweight="bold")
    axes[1,0].set_ylabel("Abundance", fontweight="bold")
    axes[2,0].set_ylabel("Difference", fontweight="bold")
    axes[3,0].set_ylabel("Density", fontweight="bold")
    axes[4,0].set_ylabel("Density", fontweight="bold")

    # Set the title for each row
    f_size = 13
    title_len = 30
    palette = [
        patch.Patch(color="royalblue", label="Codon number"),
        patch.Patch(color="goldenrod", label="Codon+GC")
    ]
    title = "Mean GC contents (%) of\nprotein-coding genes\nper species"
    axes[0,3].text(1.05, 0.25, title, fontweight="bold", fontsize=f_size,
                   transform=axes[0,3].transAxes)
    title = "Mean amino acid\nabundances (%)\nper species"
    axes[1,3].text(1.05, 0.25, title, fontweight="bold", fontsize=f_size,
                   transform=axes[1,3].transAxes)
    title = "Percentage differences (%)\nbetween empirical and\nexpected abundances\nper species"
    axes[2,3].text(1.05, 0.15, title, fontweight="bold", fontsize=f_size,
                   transform=axes[2,3].transAxes)
    title = "Spearman correlation\ncoefficients per species"
    axes[3,3].text(1.05, 0.4, title, fontweight="bold", fontsize=f_size,
                   transform=axes[3,3].transAxes)
    title = "Kendall's Tau correlation\ncoefficients per species"
    axes[4,3].text(1.05, 0.4, title, fontweight="bold", fontsize=f_size,
                   transform=axes[4,3].transAxes)

    # Set the legends for some rows
    num_text = "1 - Aliphatic    3 - Charged\n2 - Aromatic    4 - Uncharged"
    axes[1,3].text(1.08, -0.4, num_text, fontweight="bold", color="firebrick",
                   bbox=dict(facecolor="white", edgecolor="grey", alpha=0.3,
                             boxstyle="round"), linespacing=2, fontsize=10,
                   transform=axes[1,3].transAxes)
    axes[2,3].legend(handles=palette, bbox_to_anchor=(1.71, -0.1),
                     fancybox=True, fontsize=10)
    axes[3,3].legend(handles=palette, bbox_to_anchor=(1.71, -0.1),
                     fancybox=True, fontsize=10)

    # Set the plots of the first and last two rows to share the same x-axis
    # across domains
    for i in [0, 3, 4]:
        x_min = min([axes[i,j].get_xlim()[0] for j in range(4)])
        x_max = max([axes[i,j].get_xlim()[1] for j in range(4)])
        for j in range(4):
            if(i == 0):
                axes[i,j].set_xlim(x_min, x_max)
            else:
                axes[i,j].set_xlim(x_min, 1)

    # Set the plots of the next two rows to share the same y-axis across
    # domains
    for i in [1, 2]:
        y_min = min([axes[i,j].get_ylim()[0] for j in range(4)])
        y_max = max([axes[i,j].get_ylim()[1] for j in range(4)])
        for j in range(4):
            axes[i,j].set_ylim(y_min, y_max)

    # Set a grid behind all plots
    for i in range(5):
        for j in range(4):
            axes[i,j].grid(visible=True, which="major", color="#999999",
                           linestyle="dotted", alpha=0.5, zorder=0)
            axes[i,j].set_axisbelow(True)

    fig.subplots_adjust(hspace=0.5)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(input, f"plotogram.{ext}"),
                    bbox_inches="tight", transparent=True)

    plt.close()


# main method
if __name__ == "__main__":
    input = sys.argv[1]

    # Canonical amino acids order
    amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
                   "A", "G", "P", "T", "V", "L", "R", "S"]
    aa_mean_cols = [f"{aa}_mean" for aa in amino_acids]
    aa_std_cols = [f"{aa}_std" for aa in amino_acids]
    aa_groups = {"Aliphatic": ["A", "G", "I", "L", "M", "V"], "Aromatic": ["F",
                 "W", "Y"], "Charged": ["D", "E", "H", "K", "R"],
                 "Uncharged": ["C", "N", "P", "Q", "S", "T"]}

    domains = ["Archaea", "Bacteria", "Eukaryotes", "Viruses"]
    domain_corr_df = pd.DataFrame(columns=["Coefficient", "Correlation",
                                            "Comparison", "Domain"])
    all_data_dct = {}
    for domain in domains:
        king_path = os.path.join(input, domain)
        data = pd.read_csv(os.path.join(king_path, "aa_corr_results.csv"),
                           sep="\t", header=0, index_col=0)
        plot_lengths(data, domain, king_path)
        plot_gcs(data, domain, king_path)
        plot_amount(data, domain, king_path)
        plot_pct(data, aa_groups, domain, king_path)
        all_data_dct[domain] = data

    ######################################## Protein amounts across all domains
    fig,axes = plt.subplots(2, 2, sharex=True)
    i = 0
    j = 0
    amount_df = pd.DataFrame(columns=domains)
    for domain,data in all_data_dct.items():
        lengths = np.log10(data["#Proteins"]+1)
        bins = optimal_bin(lengths)
        sns.histplot(lengths, bins=bins, alpha=0.4, color="maroon", kde=True,
                     line_kws={"linewidth": 2, "linestyle": "--"}, ax=axes[i,j])
        axes[i,j].set_title(f"{domain}", fontweight="bold", fontsize=14)
        axes[i,j].set_xlabel("log10(Amount)", fontweight="bold", fontsize=12)
        axes[i,j].set_ylabel("Count", fontweight="bold", fontsize=12)
        amount_df = amount_df.reindex(np.arange(0, max(len(amount_df.index),
                                                       len(data.index))))
        amount_df[domain] = pd.Series(data["Length_mean"].values)
        j = 1 if i == 1 else j
        i = 0 if i == 1 else i + 1

    axes[0,1].set_ylabel("")
    axes[1,1].set_ylabel("")

    fig.set_figheight(10)
    fig.set_figwidth(15)
    fig.subplots_adjust(hspace=0.15, wspace=0.1)
    fig.suptitle("Distribution of protein amounts", y=0.96, fontsize=18)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(input, f"protein_amounts.{ext}"),
                    bbox_inches="tight")

    plt.close()
    amount_df.to_csv(os.path.join(input, "protein_amounts.csv"), sep="\t",
                     index=False)
    ############################################################################

    ######################################## Protein lengths across all domains
    fig,axes = plt.subplots(2, 2, sharex=True)
    i = 0
    j = 0
    length_df = pd.DataFrame(columns=domains)
    for domain,data in all_data_dct.items():
        lengths = np.log10(data["Length_mean"]+1)
        bins = optimal_bin(lengths)
        sns.histplot(lengths, bins=bins, alpha=0.4, color="maroon", kde=True,
                     line_kws={"linewidth": 2, "linestyle": "--"}, ax=axes[i,j])
        axes[i,j].set_title(f"{domain}", fontweight="bold", fontsize=14)
        axes[i,j].set_xlabel("log10(Length)", fontweight="bold", fontsize=12)
        axes[i,j].set_ylabel("Count", fontweight="bold", fontsize=12)
        length_df = length_df.reindex(np.arange(0, max(len(length_df.index),
                                                       len(data.index))))
        length_df[domain] = pd.Series(data["Length_mean"].values)
        i = 1 if j == 1 else i
        j = 0 if j == 1 else j + 1

    axes[0,1].set_ylabel("")
    axes[1,1].set_ylabel("")

    fig.set_figheight(10)
    fig.set_figwidth(15)
    fig.subplots_adjust(hspace=0.15, wspace=0.1)
    fig.suptitle("Distribution of mean protein lengths", y=0.96, fontsize=18)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(input, f"protein_lengths.{ext}"),
                    bbox_inches="tight")

    plt.close()
    length_df.to_csv(os.path.join(input, "protein_lengths.csv"), sep="\t",
                     index=False)
    ############################################################################

    #################################### Protein GC contents across all domains
    fig,axes = plt.subplots(2, 2, sharex=True)
    i = 0
    j = 0
    gcs_df = pd.DataFrame(columns=domains)
    for domain,data in all_data_dct.items():
        lengths = data["GC_mean"]
        bins = optimal_bin(lengths)
        sns.histplot(lengths, bins=bins, alpha=0.4, color="maroon", kde=True,
                     line_kws={"linewidth": 2, "linestyle": "--"}, ax=axes[i,j])
        axes[i,j].set_title(f"{domain}", fontweight="bold", fontsize=14)
        axes[i,j].set_xlabel("GC content", fontweight="bold", fontsize=12)
        axes[i,j].set_ylabel("Count", fontweight="bold", fontsize=12)
        gcs_df = gcs_df.reindex(np.arange(0, max(len(gcs_df.index),
                                                 len(data.index))))
        gcs_df[domain] = pd.Series(data["GC_mean"].values)
        i = 1 if j == 1 else i
        j = 0 if j == 1 else j + 1

    axes[0,1].set_ylabel("")
    axes[1,1].set_ylabel("")

    fig.set_figheight(10)
    fig.set_figwidth(15)
    fig.subplots_adjust(hspace=0.15, wspace=0.1)
    fig.suptitle("Distribution of mean protein GC contents", y=0.96,
                 fontsize=18)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(input, f"protein_gcs.{ext}"),
                    bbox_inches="tight")

    plt.close()
    gcs_df.to_csv(os.path.join(input, "protein_gcs.csv"), sep="\t",
                     index=False)
    ############################################################################

    ################################### Positive vs negative charged amino acids
    fig,axes = plt.subplots(2, 2, sharex=True, sharey=True)
    i = 0
    j = 0
    positve = ["D", "E"]
    negative = ["H", "K", "R"]
    charged_df = pd.DataFrame(columns=domains)
    for domain,data in all_data_dct.items():
        gcs = np.asarray((data["GC_mean"]*100).apply(np.floor)/100)
        positive_data = data[["H_mean", "K_mean", "R_mean"]].sum(axis=1)
        positive_df = pd.DataFrame(columns=["GC", "Value"])
        positive_df.loc[:,"GC"] = gcs
        positive_df.loc[:,"Value"] = list(positive_data)
        positive_df = positive_df.sort_values(by=["GC"]).astype(float)
        sns.lineplot(data=positive_df, x="GC", y="Value", errorbar=None,
                     markers=True, dashes=False, color="black", linewidth=2,
                     ax=axes[i,j])
        sns.lineplot(data=positive_df, x="GC", y="Value", errorbar="pi",
                     markers=True, dashes=False, color="royalblue", linewidth=1,
                     ax=axes[i,j])

        negative_df = pd.DataFrame(columns=["GC", "Value"])
        negative_data = data[["D_mean", "E_mean"]].sum(axis=1)
        negative_df.loc[:,"GC"] = gcs
        negative_df.loc[:,"Value"] = list(negative_data)
        negative_df = negative_df.sort_values(by=["GC"]).astype(float)
        sns.lineplot(data=negative_df, x="GC", y="Value", errorbar=None,
                     color="black", linewidth=2, ax=axes[i,j])
        sns.lineplot(data=negative_df, x="GC", y="Value", errorbar="pi",
                     color="goldenrod", linewidth=1, ax=axes[i,j])

        corr,corr_p = sci.spearmanr(positive_data, negative_data)
        axes[i,j].set_title(f"{domain}, $r_S$={corr:.3f}, p={corr_p:.3e}",
                            fontweight="bold", fontsize=14)
        axes[i,j].set_xlabel("GC content", fontweight="bold", fontsize=12)
        axes[i,j].set_ylabel("Frequency", fontweight="bold", fontsize=12)

        charged_df.loc["Positive_mean", domain] = positive_data.mean()
        charged_df.loc["Positive_std", domain] = positive_data.std()
        charged_df.loc["Negative_mean", domain] = negative_data.mean()
        charged_df.loc["Negative_std", domain] = negative_data.std()
        charged_df.loc["Spearman", domain] = corr
        charged_df.loc["Spearman_p", domain] = corr_p
        i = 1 if j == 1 else i
        j = 0 if j == 1 else j + 1

    legend_patches = [
        patch.Patch(color="royalblue", label="Positively charged AA"),
        patch.Patch(color="goldenrod", label="Negatively charged AA")
    ]
    axes[0,0].legend(handles=legend_patches, bbox_to_anchor=(1.3, -0.05),
                     fancybox=True, fontsize=12)
    axes[0,1].set_ylabel("")
    axes[1,1].set_ylabel("")

    fig.subplots_adjust(wspace=0.05, hspace=0.35)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    fig.suptitle("Frequencies of charged amino acids per GC content", y=0.96,
                 fontsize=18)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(input, f"charged_aas.{ext}"),
                    bbox_inches="tight")

    plt.close()
    charged_df.to_csv(os.path.join(input, "charged_aas.csv"), sep="\t")
    ############################################################################

    ############################# Mean amino acid abundances across all domains
    freq_df = pd.DataFrame(columns=domains, index=aa_mean_cols+aa_std_cols)
    for domain,data in all_data_dct.items():
        freq_df.loc[aa_mean_cols, domain] = data[aa_mean_cols].mean()
        freq_df.loc[aa_std_cols, domain] = data[aa_mean_cols].std().values

    freq_df.to_csv(os.path.join(input, "amino_acid_abundances.csv"), sep="\t")
    freq_df.loc[aa_mean_cols].plot(kind="bar", yerr=freq_df.loc[aa_std_cols].values.T,
                                   capsize=1, figsize=(14, 7), zorder=2)
    plt.xticks(np.arange(len(amino_acids)), amino_acids)
    plt.ylim(bottom=0)
    plt.xlabel("Amino acid", fontweight="bold", fontsize=12)
    plt.ylabel("Frequency", fontweight="bold", fontsize=12)
    plt.title("Distributions of mean amino acid frequencies across domains",
              y=1.04, fontsize=16)
    plt.legend(title="Domain", loc="upper left", fancybox=True, fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(alpha=0.5, zorder=0)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(input, f"amino_acid_distributions.{ext}"),
                    bbox_inches="tight")

    plt.close()
    ############################################################################

    #################### Percentage differences across all domains as box plots
    fig,axes = plt.subplots(2, 2, sharex=True, sharey=True)
    i = 0
    j = 0
    aa_group_order = [aa for group,aa_list in aa_groups.items()
                      for aa in aa_list]
    pct_cols = [f"{domain}_{comp_type}" for domain in domains
                                         for comp_type in ["code", "freq"]]
    pct_df = pd.DataFrame(columns=pct_cols, index=amino_acids+["Mean", "Median"])
    code_box = None
    freq_box = None
    for domain,data in all_data_dct.items():
        for comp_type in ["code", "freq"]:
            king_comp = f"{domain}_{comp_type}"
            for aa in amino_acids:
                pct_df.loc[aa, king_comp] = np.sqrt(np.mean(data[f"{aa}_pct_{comp_type}"]**2))

            pct_df.loc["Mean", king_comp] = np.mean(pct_df.loc[amino_acids, king_comp])
            pct_df.loc["Median", king_comp] = np.median(pct_df.loc[amino_acids, king_comp])

        aa_pct_cols = [f"{aa}_pct_code" for aa in aa_group_order]
        x_pos = np.arange(len(aa_pct_cols)) - 0.2
        c = "royalblue"
        code_box = axes[i,j].boxplot(data[aa_pct_cols], positions=x_pos,
                                     widths=0.3, notch=True, patch_artist=True,
                                     boxprops=dict(facecolor=c, color="black"),
                                     capprops=dict(color=c), whiskerprops=dict(color=c),
                                     flierprops=dict(color=c, markeredgecolor=c),
                                     medianprops=dict(color=c), showfliers=False,
                                     zorder=2)

        aa_pct_cols = [f"{aa}_pct_freq" for aa in aa_group_order]
        x_pos = np.arange(len(aa_pct_cols)) + 0.2
        c = "goldenrod"
        freq_box = axes[i,j].boxplot(data[aa_pct_cols], positions=x_pos,
                                     widths=0.3, notch=True, patch_artist=True,
                                     boxprops=dict(facecolor=c, color="black"),
                                     capprops=dict(color=c), whiskerprops=dict(color=c),
                                     flierprops=dict(color=c, markeredgecolor=c),
                                     medianprops=dict(color=c), showfliers=False,
                                     zorder=2)

        group_pos = 0
        last_pos = 0
        for index,(group,aa_list) in enumerate(aa_groups.items()):
            group_pos += len(aa_list)
            if(group_pos < 20):
                axes[i,j].axvline(x=group_pos-0.5, zorder=1, color="firebrick",
                                  linestyle="--", linewidth=1)

            text_pos = (group_pos+last_pos) / 20 / 2
            text = axes[i,j].text(text_pos, 0.965, index+1, color="firebrick",
                                  fontweight="bold", fontsize=12,
                                  transform=axes[i,j].transAxes)
            text.set_path_effects([
                path_effects.Stroke(linewidth=1, foreground="white"),
                path_effects.Normal()
            ])
            last_pos = group_pos

        axes[i,j].set_xticks(np.arange(len(aa_group_order)), aa_group_order)
        axes[i,j].xaxis.set_tick_params(labelbottom=True)
        axes[i,j].axhline(y=0, zorder=1, color="firebrick", linestyle="--")
        axes[i,j].grid(visible=True, which="major", color="#999999",
                       linestyle="dotted", alpha=0.5, zorder=0)
        axes[i,j].set_title(domain, fontweight="bold", fontsize=14)
        i = 1 if j == 1 else i
        j = 0 if j == 1 else j + 1

    for i in [0, 1]:
        axes[1,i].set_xlabel("Amino acid", fontweight="bold", fontsize=12)
        axes[i,0].set_ylabel("Percentage difference in %", fontweight="bold",
                             fontsize=12)

    axes[0,0].legend([code_box["boxes"][0], freq_box["boxes"][0]],
                     ["Codon number", "Codon+GC"], bbox_to_anchor=(1.05, -0.1),
                     fancybox=True, fontsize=12)
    num_text = "1 - Aliphatic    3 - Charged\n2 - Aromatic    4 - Uncharged"
    axes[0,0].text(1.05, -0.265, num_text, fontweight="bold", color="firebrick",
                   bbox=dict(facecolor="white", edgecolor="grey", alpha=0.3,
                             boxstyle="round"), linespacing=2, fontsize=12,
                   transform=axes[0,0].transAxes)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    fig.subplots_adjust(hspace=0.4, wspace=0.1)
    fig.suptitle("Percentage differences between amino acid distributions",
                 y=0.96, fontsize=18)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(input, f"amino_acid_pcts_boxplot.{ext}"),
                    bbox_inches="tight")

    plt.close()
    pct_df.to_csv(os.path.join(input, "amino_acid_pcts.csv"), sep="\t")
    ############################################################################

    ################### Percentage differences across all domains as line plots
    fig,axes = plt.subplots(2, 2, sharex=True, sharey=True)
    i = 0
    j = 0
    aa_group_order = [aa for group,aa_list in aa_groups.items()
                      for aa in aa_list]
    aa_pct_code_cols = [f"{aa}_pct_code" for aa in aa_group_order]
    aa_pct_freq_cols = [f"{aa}_pct_freq" for aa in aa_group_order]
    for domain,data in all_data_dct.items():
        pct_code_data = data[aa_pct_code_cols].melt(var_name="x", value_name="y")
        pct_code_data["z"] = ["Codon number"] * len(pct_code_data)
        pct_code_data["x"] = pct_code_data["x"].str.split("_").str[0]
        pct_freq_data = data[aa_pct_freq_cols].melt(var_name="x", value_name="y")
        pct_freq_data["z"] = ["Codon+GC"] * len(pct_freq_data)
        pct_freq_data["x"] = pct_freq_data["x"].str.split("_").str[0]
        pct_data = pd.concat([pct_code_data, pct_freq_data])

        g = sns.lineplot(data=pct_data, x="x", y="y", hue="z", style="z",
                         errorbar="pi", markers=True, dashes=False, alpha=0.8,
                         palette=["royalblue", "goldenrod"], legend=None,
                         ax=axes[i,j])

        group_pos = 0
        last_pos = 0
        for index,(group,aa_list) in enumerate(aa_groups.items()):
            group_pos += len(aa_list)
            if(group_pos < 20):
                axes[i,j].axvline(x=group_pos-0.5, zorder=1, color="firebrick",
                                  linestyle="--", linewidth=1)

            text_pos = (group_pos+last_pos) / 20 / 2
            text = axes[i,j].text(text_pos, 0.965, index+1, color="firebrick",
                                  fontweight="bold", fontsize=12,
                                  transform=axes[i,j].transAxes)
            text.set_path_effects([
                path_effects.Stroke(linewidth=1, foreground="white"),
                path_effects.Normal()
            ])
            last_pos = group_pos

        axes[i,j].xaxis.set_tick_params(labelbottom=True)
        axes[i,j].axhline(y=0, zorder=1, color="firebrick", linestyle="--")
        axes[i,j].grid(visible=True, which="major", color="#999999",
                       linestyle="dotted", alpha=0.5, zorder=0)
        axes[i,j].set_title(domain, fontweight="bold", fontsize=14)
        i = 1 if j == 1 else i
        j = 0 if j == 1 else j + 1

    for i in [0, 1]:
        axes[1,i].set_xlabel("Amino acid", fontweight="bold", fontsize=12)
        axes[i,0].set_ylabel("Percentage difference in %", fontweight="bold",
                             fontsize=12)

    legend_patches = [
        patch.Patch(color="royalblue", label="Codon number"),
        patch.Patch(color="goldenrod", label="Codon+GC")
    ]
    axes[0,0].legend(handles=legend_patches, bbox_to_anchor=(1.05, -0.1),
                     fancybox=True, fontsize=12)
    num_text = "1 - Aliphatic    3 - Charged\n2 - Aromatic    4 - Uncharged"
    axes[0,0].text(1.05, -0.265, num_text, fontweight="bold", color="firebrick",
                   bbox=dict(facecolor="white", edgecolor="grey", alpha=0.3,
                             boxstyle="round"), linespacing=2, fontsize=12,
                   transform=axes[0,0].transAxes)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    fig.subplots_adjust(hspace=0.4, wspace=0.1)
    fig.suptitle("Percentage differences between amino acid distributions",
                 y=0.96, fontsize=18)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(input, f"amino_acid_pcts_lineplot.{ext}"),
                    bbox_inches="tight")

    plt.close()
    ############################################################################

    ############## Correlation coefficients across all domains as density plots
    corr_df = pd.DataFrame(columns=["Coefficient", "Correlation", "Comparison",
                                    "Domain"])
    for domain,data in all_data_dct.items():
        for corr_type in ["Spearman", "Kendall"]:
            for comp_type in ["code", "freq"]:
                local_corr_df = pd.DataFrame(columns=["Coefficient", "Correlation",
                                                      "Comparison", "Domain"])
                local_corr_df.loc[:, "Coefficient"] = list(data[f"{corr_type}_{comp_type}"])
                local_corr_df.loc[:, "Correlation"] = [corr_type] * len(data)
                local_corr_df.loc[:, "Comparison"] = [comp_type] * len(data)
                local_corr_df.loc[:, "Domain"] = [domain] * len(data)
                corr_df = pd.concat([corr_df if not corr_df.empty else None,
                                     local_corr_df])

    corr_df.to_csv(os.path.join(input, "corr_coefficients.csv"),
                                sep="\t")
    plot_corr_coefficients(corr_df, input)
    ############################################################################

    ############### Correlation coefficients across all domains as violin plots
    g = sns.catplot(data=corr_df, x="Correlation", y="Coefficient",
                    hue="Comparison", kind="violin", col="Domain",
                    palette=["royalblue", "goldenrod"], bw_adjust=0.5, cut=0,
                    split=True, legend=None)
    g.set_axis_labels("", "")
    g.set_titles("{col_name}", fontweight="bold", size=14)
    legend_patches = [
        patch.Patch(color="royalblue", label="Codon number"),
        patch.Patch(color="goldenrod", label="Codon+GC")
    ]
    g.fig.legend(handles=legend_patches, bbox_to_anchor=(0.25, 0.9),
                 fancybox=True, fontsize=12)
    g.fig.supxlabel("Correlation type", y=0.05, fontweight="bold", fontsize=12)
    g.fig.supylabel("Correlation coefficient", x=-0.01, fontweight="bold",
                    fontsize=12)
    g.fig.set_figheight(10)
    g.fig.set_figwidth(15)
    g.fig.suptitle("Correlation coefficient distributions across domains", y=1,
                   fontsize=18)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(input, f"corr_coefficients_catplot.{ext}"),
                    bbox_inches="tight")

    plt.close()
    ############################################################################

    # All statistics together
    plot_plotogram(all_data_dct, aa_groups, input)
