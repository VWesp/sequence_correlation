import os
import sys
import numpy as np
import sympy as sp
import pandas as pd
import seaborn as sns
import textwrap as tw
import collections as col
from Bio.Data import CodonTable
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

# abbreviations for each genetic code
CODE_ABBREVIATIONS = {"AFM": "alternative_flatworm_mitochondrial", "AY": "alternative_yeast",
                      "AM": "ascidian_mitochondrial", "BAPP": "bacterial-archaeal-plant_plastid",
                      "BN": "blastocrithidia_nuclear", "CDSG": "candidate_division_sr1_and_gracilibacteria",
                      "CMUT": "cephalodiscidae_mitochondrial_uaa-tyr", "CM": "chlorophycean_mitochondrial",
                      "CDHN": "ciliate-dasycladacean-hexamita_nuclear", "CN": "condylostoma_nuclear",
                      "EFM": "echinoderm-flatworm_mitochondrial", "EN": "euplotid_nuclear",
                      "IM": "invertebrate_mitochondrial", "KN": "karyorelict_nuclear",
                      "MN": "mesodinium_nuclear", "MPCMMS": "mold-protozoan-coelenterate_mitochondrial_and_mycoplasma_spiroplasma",
                      "PTN": "pachsyolen_tannophilus_nuclear", "PN": "peritrich_nuclear",
                      "RM": "rhabdopleuridae_mitochondrial", "SOM": "scenedesmus_obliquus_mitochondrial",
                      "SGCode": "standard", "TM": "thraustochytrium_mitochondrial",
                      "TrM": "trematode_mitochondrial", "VM": "vertebrate_mitochondrial",
                      "YM": "yeast_mitochondrial"}
CODE_ABBREVIATIONS_INV = {v:k for k,v in CODE_ABBREVIATIONS.items()}

def optimal_bin(data):
    data = data.to_numpy()
    iqr = np.quantile(data, 0.75) - np.quantile(data, 0.25)
    h = 2 * iqr / len(data)**(1/3)
    return int((data.max() - data.min()) / h + 1)


if __name__ == "__main__":
    path,name = sys.argv[1:3]

    cmap = plt.get_cmap("viridis")

    # Group amino acids based on their attributes
    aa_groups = {"Aliphatic": ["A", "G", "I", "L", "M", "V"], "Aromatic": ["F",
                 "W", "Y"], "Charged": ["D", "E", "H", "K", "R"],
                 "Uncharged": ["C", "N", "P", "Q", "S", "T"]}

    prot_df = pd.read_csv(os.path.join(path, "proteome_data.csv"), sep="\t",
                            index_col=0)
    entrops = {"Proteome": np.array(prot_df["shannon_entropy"])}
    amino_acids = prot_df.columns[2:22]
    prot_mean_df = prot_df.mean(axis=0)
    prot_std_df = prot_df.std(axis=0)

    gen_cod_folders = [os.path.join(path, folder) for folder in os.listdir(path)
                       if os.path.isdir(os.path.join(path, folder))]

    cors = col.defaultdict(lambda: col.defaultdict(lambda: col.defaultdict()))
    for cod_fold in gen_cod_folders:
        code_basename = os.path.basename(cod_fold)
        # Name of the genetic code
        code_name = code_basename.capitalize().replace("_", " ")
        print(f"Current code: {code_name}...")

        cod_df = pd.read_csv(os.path.join(cod_fold, "norm_code_data.csv"),
                             sep="\t", index_col=0)
        cod_df = cod_df["frequency"]

        cor_df = pd.read_csv(os.path.join(cod_fold, "cor_data.csv"), sep="\t",
                             index_col=0)
        cors["Pearson"]["Codon"][code_name] = np.array(cor_df["codon_pear"])
        cors["Pearson"]["GC"][code_name] = np.array(cor_df["gc_pear"])
        cors["Spearman"]["Codon"][code_name] = np.array(cor_df["codon_spear"])
        cors["Spearman"]["GC"][code_name] = np.array(cor_df["gc_spear"])
        cors["log-MSE"]["Codon"][code_name] = np.array(cor_df["codon_log_mse"])
        cors["log-MSE"]["GC"][code_name] = np.array(cor_df["gc_log_mse"])

        entrops[code_name] = np.array(cor_df["shannon_entropy"])
        cor_mean_df = cor_df.mean(axis=0)
        cor_std_df = prot_df.std(axis=0)

        fig, axes = plt.subplots(2, 2)
        i = 0
        j = 0
        for aa_type,aa_list in aa_groups.items():
            x_data = np.arange(len(aa_list))
            wid = 0.25
            b_pos = -0.25
            col = 0.3
            # Plot observed mean relative abundance and standard deviation of
            # each amino acid
            axes[i,j].bar(x_data+b_pos, prot_mean_df[aa_list],
                          yerr=prot_std_df[aa_list], width=wid, color=cmap(col),
                          edgecolor="black", linewidth=0.75, label="Observed",
                          capsize=3, zorder=2)

            cor_data = {"Codon": [cod_df, None, "codon"],
                        "Codon+GC": [cor_mean_df, cor_std_df, "gc"]}
            c_pos = 0.6
            for cor_type,data in cor_data.items():
                b_pos += wid
                col += 0.3
                yerr = data[1][aa_list] if data[1] is not None else None
                axes[i,j].bar(x_data+b_pos, data[0][aa_list],
                              yerr=yerr, width=wid, color=cmap(col),
                              edgecolor="black", label=cor_type,
                              linewidth=0.75, capsize=3, zorder=2)

                axes[i,j].grid(visible=True, which="major", color="#999999",
                               linestyle="dotted", alpha=0.5, zorder=0)

                axes[i,j].set_xticks(np.arange(len(aa_list)), aa_list)
                axes[i,j].set_xlabel("Amino acid")
                axes[i,j].set_title(f"{aa_type} amino acids")
                if(j == 0):
                    axes[i,j].set_ylabel("Mean amino acid frequency")
                    if(i == 0):
                        pear_ar = [f"{data[2]}_pear",
                                   f"p_{data[2]}_pear"]
                        pcc,pcc_p = cor_mean_df[pear_ar]

                        spear_ar = [f"{data[2]}_spear",
                                    f"p_{data[2]}_spear"]
                        r2, r2_p = cor_mean_df[spear_ar]

                        mse = cor_mean_df[f"{data[2]}_log_mse"]

                        label = tw.fill(f"${cor_type}$ correlation:", 30)
                        axes[i,j].text(1.03, c_pos, f"{label}\n"
                            f"  - Mean Pearson:\n"
                            f"    - Coefficient: {pcc:.5f}\n"
                            f"    - p-value: {pcc_p:.5e}\n"
                            f"\n  - Mean Spearman:\n"
                            f"    - Coefficient: {r2:.5f}\n"
                            f"    - p-value: {r2_p:.5e}",
                            transform=axes[0,0].transAxes, fontsize=11,
                            verticalalignment="top", linespacing=1.5,
                            bbox=dict(boxstyle="round", facecolor="white",
                                      edgecolor="grey", alpha=0.5))
                c_pos -= 0.7

            j = 1 if i == 1 else j
            i = 0 if i == 1 else i + 1

        y_max = max(max([ax.get_ylim() for ax in axes.reshape(-1)]))
        for ax in axes.reshape(-1):
            ax.set_ylim(0, y_max)

        axes[0,0].legend(loc="upper center", bbox_to_anchor=(1.19, 1.02),
                         fancybox=True, fontsize=12)

        fig.subplots_adjust(wspace=0.6, hspace=0.3)
        title = tw.fill(f"{name} - Mean proteome amino acid frequency "
                        f"for genetic code: {code_name}", 100)
        fig.suptitle(title, fontsize=15, y=0.95)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        for ext in ["svg", "pdf"]:
            plt.savefig(f"{cod_fold}/cor_bar_plot.{ext}", bbox_inches="tight")

        plt.close()

        for type in ["Pearson", "Spearman"]:
            joint_df = pd.DataFrame({"GC": prot_df["GC"],
                                     "Codon_Cor": cors[type]["Codon"][code_name],
                                     "GC_Cor": cors[type]["GC"][code_name]})
            joint_df["Cor_Dif"] = np.abs(joint_df["Codon_Cor"]-joint_df["GC_Cor"])
            g = sns.JointGrid(data=joint_df, x="GC", y="Codon_Cor")
            g.plot_joint(sns.scatterplot, alpha=0.5, color="blue",
                         label="Codon correlation")
            sns.scatterplot(data=joint_df, x="GC", y="GC_Cor", alpha=0.5,
                            color="red", ax=g.ax_joint, label="Codon+GC correlation")

            sns.histplot(data=joint_df["GC"], ax=g.ax_marg_x, color="purple",
                         element="step")

            hist_y1, bins_y1 = np.histogram(joint_df["Codon_Cor"], density=True,
                                            bins=optimal_bin(joint_df["Codon_Cor"]))
            g.ax_marg_y.fill_betweenx(bins_y1[:-1], 0, hist_y1, step="pre",
                                      color="blue", alpha=0.5)

            hist_y2, bins_y2 = np.histogram(joint_df["GC_Cor"], density=True,
                                            bins=optimal_bin(joint_df["GC_Cor"]))
            g.ax_marg_y.fill_betweenx(bins_y2[:-1], 0, hist_y2, step="pre",
                                      color="red", alpha=0.5)

            g.ax_joint.set_xlabel("GC content")
            g.ax_joint.set_ylabel(f"{type} correlation coefficient")
            title = tw.fill(f"{name} - {type} correlation coefficients "
                            f"for genetic code: {code_name}", 100)
            g.fig.suptitle(title)
            sns.move_legend(g.ax_joint, "upper left")
            g.fig.subplots_adjust(top=0.92)
            g.fig.set_figheight(10)
            g.fig.set_figwidth(15)

            for ext in ["svg", "pdf"]:
                plt.savefig(f"{cod_fold}/cor_{type.lower()}_scatter_plot.{ext}",
                            bbox_inches="tight")

            plt.close()

    print("Plotting remaining stuff...", end="\n\n")

    # create the colors
    cmap = plt.get_cmap("gist_rainbow")

    for cor_type in ["Pearson", "Spearman"]:
        fig, axes = plt.subplots(len(gen_cod_folders), 2, figsize=(10, 10))
        index = 0
        code_abbr = []
        for code,data in cors[cor_type]["Codon"].items():
            code_basename = code.lower().replace(" ", "_")
            code_abbr.append(CODE_ABBREVIATIONS_INV[code_basename])

            color = cmap(index/len(gen_cod_folders))
            if(code == "Standard"):
                color = "grey"

            sns.violinplot(x=data, ax=axes[index,0], zorder=2, color=color)
            axes[index,0].hlines(0, -1, np.min(data), color="black",
                                 linestyle="--", alpha=0.5, zorder=0)
            axes[index,0].hlines(0, np.max(data), 1, color="black",
                                 linestyle="--", alpha=0.5, zorder=0)
            axes[index,0].set_xlim(-1, 1)
            axes[index,0].set_yticks([])
            for spine in axes[index,0].spines.values():
                spine.set_visible(False)

            if(index != len(gen_cod_folders)-1):
                axes[index,0].set_xticks([])
            else:
                axes[index,0].set_xticks([-1, -0.5, 0, 0.5, 1])

            if(index == 0):
                axes[index,0].set_title("Codon")

            index += 1

        # Add a large border around the left set of plots
        left_rect = patch.Rectangle(
            (0.12, 0.11), 0.36, 0.77,
            linewidth=2, edgecolor="black", facecolor="None",
            transform=fig.transFigure, clip_on=False
        )
        fig.patches.append(left_rect)

        # Add a continuous vertical line across all left-side subplots
        left_line = Line2D(
            [0.3013, 0.3013], [0.11, 0.881],
            color="black", linestyle="--", linewidth=1.5, transform=fig.transFigure,
            clip_on=False, alpha=0.7, zorder=0
        )
        fig.lines.append(left_line)

        index = 0
        for code,data in cors[cor_type]["GC"].items():
            color = cmap(index/len(gen_cod_folders))
            if(code == "Standard"):
                color = "grey"

            sns.violinplot(x=data, ax=axes[index,1], zorder=2, color=color)
            axes[index,1].hlines(0, -1, np.min(data), color="black",
                                 linestyle="--", alpha=0.5, zorder=0)
            axes[index,1].hlines(0, np.max(data), 1, color="black",
                                 linestyle="--", alpha=0.5, zorder=0)
            axes[index,1].set_xlim(-1, 1)
            axes[index,1].set_yticks([])
            for spine in axes[index,1].spines.values():
                spine.set_visible(False)

            if(index != len(gen_cod_folders)-1):
                axes[index,1].set_xticks([])
            else:
                axes[index,1].set_xticks([-1, -0.5, 0, 0.5, 1])

            if(index == 0):
                axes[index,1].set_title("Codon+GC")

            index += 1

        # Add a large border around the right set of plots
        right_rect = patch.Rectangle(
            (0.544, 0.11), 0.36, 0.77,
            linewidth=2, edgecolor="black", facecolor="none",
            transform=fig.transFigure, clip_on=False
        )
        fig.patches.append(right_rect)

        # Add a continuous vertical line across all right-side subplots
        right_line = Line2D(
            [0.7236, 0.7236], [0.11, 0.881],
            color="black", linestyle="--", linewidth=1.5, transform=fig.transFigure,
            clip_on=False, alpha=0.7, zorder=0
        )
        fig.lines.append(right_line)

        dist = 0.0311
        for i,label in enumerate(code_abbr):
            pos = 0.1
            if(len(label) == 3):
                pos = 0.092
            elif(len(label) == 4):
                pos = 0.084
            elif(len(label) == 6):
                pos = 0.0675

            fig.text(pos, 0.865-dist*i, label, fontdict={"family": "monospace"})

        fig.suptitle(f"{name} - {cor_type} correlation coefficients between all genetic codes",
                     fontsize=14, y=0.93)

        for ext in ["svg", "pdf"]:
            plt.savefig(f"{path}/{cor_type}_cor_plots.{ext}",
                        bbox_inches="tight")

        plt.close()

    ###############################################################
    fig, axes = plt.subplots(len(gen_cod_folders), 2, figsize=(10, 10))
    index = 0
    max_val_cod = max([max(data) for _,data in cors["log-MSE"]["Codon"].items()])
    max_val_gc = max([max(data) for _,data in cors["log-MSE"]["GC"].items()])
    max_value = max(max_val_cod*1.15, max_val_gc*1.15)
    code_abbr = []
    for code,data in cors["log-MSE"]["Codon"].items():
        code_basename = code.lower().replace(" ", "_")
        code_abbr.append(CODE_ABBREVIATIONS_INV[code_basename])

        color = cmap(index/len(gen_cod_folders))
        if(code == "Standard"):
            color = "grey"

        sns.violinplot(x=data, ax=axes[index,0], zorder=2, color=color)
        axes[index,0].hlines(0, 0, np.min(data), color="black",
                             linestyle="--", alpha=0.5, zorder=0)
        axes[index,0].hlines(0, np.max(data), max_value, color="black",
                             linestyle="--", alpha=0.5, zorder=0)
        axes[index,0].set_xlim(0, max_value)
        axes[index,0].set_yticks([])
        for spine in axes[index,0].spines.values():
            spine.set_visible(False)

        if(index != len(gen_cod_folders)-1):
            axes[index,0].set_xticks([])

        if(index == 0):
            axes[index,0].set_title("Codon")

        index += 1

    # Add a large border around the left set of plots
    left_rect = patch.Rectangle(
        (0.12, 0.11), 0.36, 0.77,
        linewidth=2, edgecolor="black", facecolor="None",
        transform=fig.transFigure, clip_on=False
    )
    fig.patches.append(left_rect)

    index = 0
    for code,data in cors["log-MSE"]["GC"].items():
        color = cmap(index/len(gen_cod_folders))
        if(code == "Standard"):
            color = "grey"

        sns.violinplot(x=data, ax=axes[index,1], zorder=2, color=color)
        axes[index,1].hlines(0, 0, np.min(data), color="black",
                             linestyle="--", alpha=0.5, zorder=0)
        axes[index,1].hlines(0, np.max(data), max_value, color="black",
                             linestyle="--", alpha=0.5, zorder=0)
        axes[index,1].set_xlim(0, max_value)
        axes[index,1].set_yticks([])
        for spine in axes[index,1].spines.values():
            spine.set_visible(False)

        if(index != len(gen_cod_folders)-1):
            axes[index,1].set_xticks([])

        if(index == 0):
            axes[index,1].set_title("Codon+GC")

        index += 1

    # Add a large border around the right set of plots
    right_rect = patch.Rectangle(
        (0.544, 0.11), 0.36, 0.77,
        linewidth=2, edgecolor="black", facecolor="none",
        transform=fig.transFigure, clip_on=False
    )
    fig.patches.append(right_rect)

    dist = 0.0311
    for i,label in enumerate(code_abbr):
        pos = 0.1
        if(len(label) == 3):
            pos = 0.092
        elif(len(label) == 4):
            pos = 0.084
        elif(len(label) == 6):
            pos = 0.0675

        fig.text(pos, 0.865-dist*i, label, fontdict={"family": "monospace"})

    fig.suptitle(f"{name} - log-MSE values between all genetic codes",
                 fontsize=14, y=0.93)

    for ext in ["svg", "pdf"]:
        plt.savefig(f"{path}/log_mse_plot.{ext}", bbox_inches="tight")

    plt.close()

    ###############################################################
    fig, axes = plt.subplots(len(gen_cod_folders)+1, 1, figsize=(10, 10))
    index = 0
    min_value = min([min(data) for _,data in entrops.items()]) * 0.95
    max_value = max([max(data) for _,data in entrops.items()]) * 1.05
    code_abbr = []
    for code,data in entrops.items():
        code_name = None
        if(index == 0):
            code_abbr.append("Data")
        else:
            code_basename = code.lower().replace(" ", "_")
            code_abbr.append(CODE_ABBREVIATIONS_INV[code_basename])

        color = cmap((index-1)/len(gen_cod_folders))
        if(index == 0 or code == "Standard"):
            color = "grey"

        sns.violinplot(x=data, ax=axes[index], zorder=2, color=color)
        axes[index].hlines(0, min_value, np.min(data), color="black",
                             linestyle="--", alpha=0.5, zorder=0)
        axes[index].hlines(0, np.max(data), max_value, color="black",
                             linestyle="--", alpha=0.5, zorder=0)
        axes[index].set_xlim(min_value, max_value)
        axes[index].set_yticks([])
        for spine in axes[index].spines.values():
            spine.set_visible(False)

        if(index != len(gen_cod_folders)):
            axes[index].set_xticks([])

        if(index == 0):
            axes[index].set_title(f"{name} - Shannon entropy based on codon number and GC content")

        index += 1

    # Add a large border around the set of plots
    rect = patch.Rectangle(
        (0.12, 0.11), 0.78, 0.77,
        linewidth=2, edgecolor="black", facecolor="None",
        transform=fig.transFigure, clip_on=False
    )
    fig.patches.append(rect)

    dist = 0.0298
    for i,label in enumerate(code_abbr):
        pos = 0.1
        if(len(label) == 3):
            pos = 0.092
        elif(len(label) == 4):
            pos = 0.084
        elif(len(label) == 6):
            pos = 0.0675

        fig.text(pos, 0.865-dist*i, label, fontdict={"family": "monospace"})

    for ext in ["svg", "pdf"]:
        plt.savefig(f"{path}/shannon_plot.{ext}", bbox_inches="tight")

    plt.close()

    ###############################################################
    data = prot_df["Length"]
    bins = optimal_bin(data)
    sns.histplot(data, bins=bins, alpha=0.4, color="maroon", kde=True,
                 line_kws={"linewidth": 2, "linestyle": "--"})
    plt.title(f"{name} - Density of median proteome length")
    plt.xlabel("Protein length")
    plt.ylabel("Count")

    for ext in ["svg", "pdf"]:
        plt.savefig(f"{path}/length_plot.{ext}", bbox_inches="tight")

    plt.close()

    ###############################################################
    data = prot_df["GC"]
    bins = optimal_bin(data)
    sns.histplot(data, bins=bins, alpha=0.4, color="maroon", kde=True,
                 line_kws={"linewidth": 2, "linestyle": "--"})
    plt.title(f"{name} - Density of median proteome GC content")
    plt.xlabel("GC content")
    plt.ylabel("Count")

    for ext in ["svg", "pdf"]:
        plt.savefig(f"{path}/gc_plot.{ext}", bbox_inches="tight")

    plt.close()
