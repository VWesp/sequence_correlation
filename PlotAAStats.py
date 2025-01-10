import os
import sys
import numpy as np
import sympy as sp
import pandas as pd
import seaborn as sns
import textwrap as tw
import collections as defcol
from scipy.stats import norm
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
    path = sys.argv[1]

    # Group amino acids based on their attributes
    aa_groups = {"Aliphatic": ["A", "G", "I", "L", "M", "V"], "Aromatic": ["F",
                 "W", "Y"], "Charged": ["D", "E", "H", "K", "R"],
                 "Uncharged": ["C", "N", "P", "Q", "S", "T"]}

    kingdoms = ["Archaea", "Bacteria", "Eukaryota", "Viruses"]
    king_freq_data = defcol.defaultdict(lambda: defcol.defaultdict(lambda:{}))
    standard_norm_data = None
    standard_freq_data = {}
    amino_acids = None
    z_corr = defcol.defaultdict(lambda: defcol.defaultdict(lambda: defcol.defaultdict(lambda: defcol.defaultdict(lambda: defcol.defaultdict()))))
    for kingdom in kingdoms:
        cmap = plt.get_cmap("viridis")
        print(f"Current kingdom: {kingdom}...")
        king_path = os.path.join(path, kingdom.lower())
        prot_df = pd.read_csv(os.path.join(king_path, "proteome_data.csv"), sep="\t",
                                index_col=0)
        amino_acids = prot_df.columns[3:23]
        num_prots = prot_df["#Proteins"]
        z_weights = [n-3 for n in num_prots]
        weighted_means = prot_df.mul(num_prots, axis=0).sum() / num_prots.sum()
        weighted_std = np.sqrt(((prot_df-weighted_means)**2).mul(num_prots, axis=0).sum() / num_prots.sum())
        entrops = {"Proteome": np.array(prot_df["shannon_entropy"])}

        king_freq_data["means"][kingdom] = {aa:weighted_means[aa] for aa in amino_acids}
        king_freq_data["std"][kingdom] = {aa:weighted_std[aa] for aa in amino_acids}

        gen_cod_folders = [os.path.join(king_path, folder) for folder in os.listdir(king_path)
                           if os.path.isdir(os.path.join(king_path, folder))]
        corr = defcol.defaultdict(lambda: defcol.defaultdict(lambda: defcol.defaultdict()))
        for cod_fold in gen_cod_folders:
            code_basename = os.path.basename(cod_fold)
            # Name of the genetic code
            code_name = code_basename.capitalize().replace("_", " ")
            print(f"\tCurrent code: {code_name}...")

            cod_df = pd.read_csv(os.path.join(cod_fold, "norm_code_data.csv"),
                                 sep="\t", index_col=0)
            cod_df = cod_df["frequency"]

            corr_df = pd.read_csv(os.path.join(cod_fold, "corr_data.csv"), sep="\t",
                                 index_col=0)
            corr["Pearson"]["Codon"][code_name] = np.array(corr_df["codon_pear"])
            corr["Pearson"]["GC"][code_name] = np.array(corr_df["gc_pear"])
            corr["Spearman"]["Codon"][code_name] = np.array(corr_df["codon_spear"])
            corr["Spearman"]["GC"][code_name] = np.array(corr_df["gc_spear"])
            corr["log-MSE"]["Codon"][code_name] = np.array(corr_df["codon_log_mse"])
            corr["log-MSE"]["GC"][code_name] = np.array(corr_df["gc_log_mse"])
            entrops[code_name] = np.array(corr_df["shannon_entropy"])

            for type in ["Codon", "GC"]:
                z_values = [0.5*np.log((1+corr)/(1-corr)) for corr in corr["Pearson"][type][code_name]]
                z_mean = np.average(z_values, weights=z_weights)
                z_corr[kingdom]["Pearson"][type][code_name]["means"] = (np.exp(2*z_mean)-1)/(np.exp(2*z_mean)+1)
                z_corr[kingdom]["Pearson"][type][code_name]["std"] = 1 / np.sqrt(np.sum(z_weights))
                z_stat = z_mean / z_corr[kingdom]["Pearson"][type][code_name]["std"]
                z_corr[kingdom]["Pearson"][type][code_name]["p"] = 2 * (1 - norm.cdf(abs(z_stat)))

            for type in ["Codon", "GC"]:
                z_values = [0.5*np.log((1+corr)/(1-corr)) for corr in corr["Spearman"][type][code_name]]
                z_mean = np.average(z_values, weights=z_weights)
                z_corr[kingdom]["Spearman"][type][code_name]["means"] = (np.exp(2*z_mean)-1)/(np.exp(2*z_mean)+1)
                z_corr[kingdom]["Spearman"][type][code_name]["std"] = 1 / np.sqrt(np.sum(z_weights))
                z_stat = z_mean / z_corr[kingdom]["Spearman"][type][code_name]["std"]
                z_corr[kingdom]["Spearman"][type][code_name]["p"] = 2 * (1 - norm.cdf(abs(z_stat)))

            pred_freq_df = pd.read_csv(os.path.join(cod_fold, "pred_freq_data.csv"), sep="\t",
                                       index_col=0)

            if(code_name == "Standard"):
                standard_norm_data = cod_df
                standard_freq_data[kingdom] = corr_df

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
                axes[i,j].bar(x_data+b_pos, weighted_means[aa_list], yerr=weighted_std[aa_list],
                              width=wid, color=cmap(col), edgecolor="black", linewidth=0.75,
                              label="Observed", capsize=3, zorder=2)

                corr_data = {"Codon": [cod_df, None, "codon"],
                            "Codon+GC": [pred_freq_df.mean(axis=0), pred_freq_df.std(axis=0), "gc"]}
                c_pos = 0.6
                for corr_type,data in corr_data.items():
                    b_pos += wid
                    col += 0.3
                    yerr = data[1][aa_list] if data[1] is not None else None
                    axes[i,j].bar(x_data+b_pos, data[0][aa_list],
                                  yerr=yerr, width=wid, color=cmap(col),
                                  edgecolor="black", label=corr_type,
                                  linewidth=0.75, capsize=3, zorder=2)

                    axes[i,j].grid(visible=True, which="major", color="#999999",
                                   linestyle="dotted", alpha=0.5, zorder=0)

                    axes[i,j].set_xticks(np.arange(len(aa_list)), aa_list)
                    axes[i,j].set_xlabel("Amino acid")
                    axes[i,j].set_title(f"{aa_type} amino acids")
                    if(j == 0):
                        axes[i,j].set_ylabel("Mean amino acid frequency")
                        if(i == 0):
                            seq_type = "Codon" if corr_type=="Codon" else "GC"
                            pearsons = z_corr[kingdom]["Pearson"][seq_type][code_name]
                            spearmans = z_corr[kingdom]["Spearman"][seq_type][code_name]
                            label = tw.fill(f"${corr_type}$ correlation:", 30)
                            axes[i,j].text(1.03, c_pos, f"{label}\n"
                                f"  - Pearson:\n"
                                f"    - Mean coef.: {pearsons["means"]:.5f}\n"
                                f"    - Std coef.: {pearsons["std"]:.5f}\n"
                                f"    - p-value: {pearsons["p"]:.3f}\n"
                                f"\n  - Spearman:\n"
                                f"    - Mean coef.: {spearmans["means"]:.5f}\n"
                                f"    - Std coef.: {spearmans["std"]:.5f}\n"
                                f"    - p-value: {spearmans["p"]:.3f}",
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
            title = tw.fill(f"{kingdom} - Mean proteome amino acid frequency "
                            f"for genetic code: {code_name}", 100)
            fig.suptitle(title, fontsize=15, y=0.95)
            fig.set_figheight(10)
            fig.set_figwidth(15)
            for ext in ["svg", "pdf"]:
                plt.savefig(f"{cod_fold}/corr_bar_plot.{ext}", bbox_inches="tight")

            plt.close()

            for type in ["Pearson", "Spearman"]:
                joint_df = pd.DataFrame({"GC": prot_df["GC"],
                                         "Codon_corr": corr[type]["Codon"][code_name],
                                         "GC_corr": corr[type]["GC"][code_name]})
                joint_df["corr_Dif"] = np.abs(joint_df["Codon_corr"]-joint_df["GC_corr"])
                g = sns.JointGrid(data=joint_df, x="GC", y="Codon_corr")
                g.plot_joint(sns.scatterplot, alpha=0.5, color="blue",
                             label="Codon correlation")
                sns.scatterplot(data=joint_df, x="GC", y="GC_corr", alpha=0.5,
                                color="red", ax=g.ax_joint, label="Codon+GC correlation")

                sns.histplot(data=joint_df["GC"], ax=g.ax_marg_x, color="purple",
                             element="step")

                hist_y1, bins_y1 = np.histogram(joint_df["Codon_corr"], density=True,
                                                bins=optimal_bin(joint_df["Codon_corr"]))
                g.ax_marg_y.fill_betweenx(bins_y1[:-1], 0, hist_y1, step="pre",
                                          color="blue", alpha=0.5)

                hist_y2, bins_y2 = np.histogram(joint_df["GC_corr"], density=True,
                                                bins=optimal_bin(joint_df["GC_corr"]))
                g.ax_marg_y.fill_betweenx(bins_y2[:-1], 0, hist_y2, step="pre",
                                          color="red", alpha=0.5)

                g.ax_joint.set_xlabel("GC content")
                g.ax_joint.set_ylabel(f"{type} correlation coefficient")
                title = tw.fill(f"{kingdom} - {type} correlation coefficients "
                                f"for genetic code: {code_name}", 100)
                g.fig.suptitle(title)
                sns.move_legend(g.ax_joint, "lower right")
                g.fig.subplots_adjust(top=0.92)
                g.fig.set_figheight(10)
                g.fig.set_figwidth(15)

                for ext in ["svg", "pdf"]:
                    plt.savefig(f"{cod_fold}/corr_{type.lower()}_scatter_plot.{ext}",
                                bbox_inches="tight")

                plt.close()

        print("\tPlotting remaining stuff...", end="\n\n")

        # create the colors
        cmap = plt.get_cmap("gist_rainbow")

        for corr_type in ["Pearson", "Spearman"]:
            fig, axes = plt.subplots(len(gen_cod_folders), 2, figsize=(10, 10))
            index = 0
            code_abbr = []
            for code,data in corr[corr_type]["Codon"].items():
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
            for code,data in corr[corr_type]["GC"].items():
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

            fig.suptitle(f"{kingdom} - {corr_type} correlation coefficients between all genetic codes",
                         fontsize=14, y=0.93)

            for ext in ["svg", "pdf"]:
                plt.savefig(f"{king_path}/{corr_type}_corr_plots.{ext}",
                            bbox_inches="tight")

            plt.close()

        ###############################################################
        fig, axes = plt.subplots(len(gen_cod_folders), 2, figsize=(10, 10))
        index = 0
        max_val_cod = max([max(data) for _,data in corr["log-MSE"]["Codon"].items()])
        max_val_gc = max([max(data) for _,data in corr["log-MSE"]["GC"].items()])
        max_value = max(max_val_cod*1.15, max_val_gc*1.15)
        code_abbr = []
        for code,data in corr["log-MSE"]["Codon"].items():
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
        for code,data in corr["log-MSE"]["GC"].items():
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

        fig.suptitle(f"{kingdom} - log-MSE values between all genetic codes",
                     fontsize=14, y=0.93)

        for ext in ["svg", "pdf"]:
            plt.savefig(f"{king_path}/log_mse_plot.{ext}", bbox_inches="tight")

        plt.close()

        ###############################################################
        fig, axes = plt.subplots(len(gen_cod_folders)+1, 1, figsize=(10, 10))
        index = 0
        min_value = min([min(data) for _,data in entrops.items()]) * 0.98
        max_value = max([max(data) for _,data in entrops.items()]) * 1.02
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
                axes[index].set_title(f"{kingdom} - Shannon entropies based on codon number and GC content")

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
            plt.savefig(f"{king_path}/shannon_plot.{ext}", bbox_inches="tight")

        plt.close()

        ###############################################################
        data = prot_df["Length"]
        bins = optimal_bin(data)
        sns.histplot(data, bins=bins, alpha=0.4, color="maroon", kde=True,
                     line_kws={"linewidth": 2, "linestyle": "--"})
        plt.title(f"{kingdom} - Density of mean proteome length")
        plt.xlabel("Protein length")
        plt.ylabel("Count")

        for ext in ["svg", "pdf"]:
            plt.savefig(f"{king_path}/length_plot.{ext}", bbox_inches="tight")

        plt.close()

        ###############################################################
        data = prot_df["GC"]
        bins = optimal_bin(data)
        sns.histplot(data, bins=bins, alpha=0.4, color="maroon", kde=True,
                     line_kws={"linewidth": 2, "linestyle": "--"})
        plt.title(f"{kingdom} - Density of mean proteome GC content")
        plt.xlabel("GC content")
        plt.ylabel("Count")

        for ext in ["svg", "pdf"]:
            plt.savefig(f"{king_path}/gc_plot.{ext}", bbox_inches="tight")

        plt.close()

    ###############################################################
    king_freq_mean_df = pd.DataFrame(king_freq_data["means"])
    king_freq_std_df = pd.DataFrame(king_freq_data["std"])
    king_freq_mean_df.plot(kind="bar", yerr=king_freq_std_df, capsize=1,
                        figsize=(14, 7), zorder=2)
    plt.xlabel("Amino acids")
    plt.ylabel("Mean distribution")
    plt.title("Mean proteomic amino acid distribution across kingdoms")
    plt.legend(title="Kingdom", loc="upper left")
    plt.xticks(rotation=0)
    plt.grid(alpha=0.5, zorder=0)
    for ext in ["svg", "pdf"]:
        plt.savefig(f"{os.path.join(path, "standard_mean_amino_acid_distributions")}.{ext}",
                    bbox_inches="tight")

    plt.close()

    standard_freq_mean_df = pd.DataFrame.from_dict(standard_freq_data).loc[amino_acids]

    all_king_data = pd.DataFrame(columns=amino_acids)
    all_king_data.loc["Code frequency"] = np.array(standard_norm_data)
    weighted_indices = []
    code_diff_indices = []
    pred_diff_indices = []
    for kingdom in kingdoms:
        weighted_indices.append(f"{kingdom} observed mean frequency")
        code_diff_indices.append(f"{kingdom} code percentage difference")
        pred_diff_indices.append(f"{kingdom} predicted percentage difference")
        all_king_data.loc[f"{kingdom} observed mean frequency"] = king_freq_mean_df[kingdom]
        all_king_data.loc[f"{kingdom} observed std frequency"] = king_freq_std_df[kingdom]
        all_king_data.loc[f"{kingdom} predicted mean frequency"] = standard_freq_mean_df[kingdom]
        all_king_data.loc[f"{kingdom} code percentage difference"] = all_king_data.loc[[f"{kingdom} observed mean frequency", "Code frequency"]].pct_change().iloc[1]
        all_king_data.loc[f"{kingdom} predicted percentage difference"] = all_king_data.loc[[f"{kingdom} observed mean frequency", f"{kingdom} predicted  mean frequency"]].pct_change().iloc[1]

    all_king_data.loc["Mean frequency"] = all_king_data.loc[weighted_indices].mean(axis=0)
    all_king_data.loc["Std frequency"] = all_king_data.loc[weighted_indices].std(axis=0)
    all_king_data.loc["Minimum frequency"] = all_king_data.loc[weighted_indices].min(axis=0)
    all_king_data.loc["Maximum frequency"] = all_king_data.loc[weighted_indices].max(axis=0)

    all_king_data.loc["Mean code percentage difference"] = np.mean(np.absolute(all_king_data.loc[code_diff_indices]))
    all_king_data.loc["Mean predicted percentage difference"] = np.mean(np.absolute(all_king_data.loc[pred_diff_indices]))

    all_king_data["Mean"] = all_king_data[amino_acids].mean(axis=1)
    all_king_data["Minimum"] = all_king_data[amino_acids].min(axis=1)
    all_king_data["Maximum"] = all_king_data[amino_acids].max(axis=1)

    all_king_data.to_csv(os.path.join(path, "standard_mean_amino_acid_distributions.csv"), sep="\t")
