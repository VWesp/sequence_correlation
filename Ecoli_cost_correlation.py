import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as sci
import seaborn as sns
import matplotlib.pyplot as plt


def s_corr_permut_test(x, y, permuts):
    rank_x = sci.rankdata(x)
    rank_y = sci.rankdata(y)
    real_corr, _ = sci.spearmanr(rank_x, rank_y)
    permuted_corrs = np.array([
        sci.spearmanr(np.random.permutation(rank_x), rank_y)[0]
        for _ in range(permuts)
    ])
    real_p_value = np.mean(np.abs(permuted_corrs) >= np.abs(real_corr))
    return [real_corr, real_p_value]


if __name__ == '__main__':
    bacteria = pd.read_csv(sys.argv[1], sep="\t", header=0, index_col=0)
    ecoli = bacteria.loc["UP000000625_83333"]

    amino_acids = ["M_mean", "W_mean", "C_mean", "D_mean", "E_mean", "F_mean",
                   "H_mean", "K_mean", "N_mean", "Q_mean", "Y_mean", "I_mean",
                   "A_mean", "G_mean", "P_mean", "T_mean", "V_mean", "L_mean",
                   "R_mean", "S_mean"]
    ecoli_aa_freqs = ecoli[amino_acids]
    ecoli_aa_glocuse = [18, -1, 8, 0,-7, 0, 3, 5, 2, -6, -2, 7, -1, -2, -2, 6,
                        -2, -9, 0, -2]
    ecoli_aa_gylcerol = [16, -2, 6, -2, -11, 4.33, 4.33, 1, 0, -10, 6.33, 3, -3,
                         -4, -6, 4, -6, -15, -4, -4]
    ecoli_aa_acetate = [17, 6, 8, -1, -2, 2.33, 7.67, 4, 1, -1, 0.33, 6, -1, -2,
                        3, 5, -2, -5, 5, -2]

    corr_cost_df = pd.DataFrame(columns=["Corr", "P-value", "Synthesis", "Correlation test"],
                                index=[np.arange(0, 6, 1)])
    repeats = 100000
    spear_glucose_corr = s_corr_permut_test(ecoli_aa_freqs.values, ecoli_aa_glocuse,
                                            repeats)
    corr_cost_df.iloc[0] = [spear_glucose_corr[0], spear_glucose_corr[1],
                            "Glucose", "Spearman rank"]
    spear_glycerol_corr = s_corr_permut_test(ecoli_aa_freqs.values, ecoli_aa_gylcerol,
                                            repeats)
    corr_cost_df.iloc[1] = [spear_glycerol_corr[0], spear_glycerol_corr[1],
                            "Gylcerol", "Spearman rank"]
    spear_acetate_corr = s_corr_permut_test(ecoli_aa_freqs.values, ecoli_aa_acetate,
                                            repeats)
    corr_cost_df.iloc[2] = [spear_acetate_corr[0], spear_acetate_corr[1],
                            "Acetate", "Spearman rank"]

    kendall_glucose_corr = sci.kendalltau(ecoli_aa_freqs.values, ecoli_aa_glocuse,
                                           nan_policy="raise")
    corr_cost_df.iloc[3] = [kendall_glucose_corr[0], kendall_glucose_corr[1],
                            "Glucose", "Kendall's tau"]
    kendall_glycerol_corr = sci.kendalltau(ecoli_aa_freqs.values, ecoli_aa_gylcerol,
                                            nan_policy="raise")
    corr_cost_df.iloc[4] = [kendall_glycerol_corr[0], kendall_glycerol_corr[1],
                            "Gylcerol", "Kendall's tau"]
    kendall_acetate_corr = sci.kendalltau(ecoli_aa_freqs.values, ecoli_aa_acetate,
                                           nan_policy="raise")
    corr_cost_df.iloc[5] = [kendall_acetate_corr[0], kendall_acetate_corr[1],
                            "Acetate", "Kendall's tau"]

    sns.barplot(corr_cost_df, x="Synthesis", y="Corr", hue="Correlation test",
                palette=["royalblue", "goldenrod"], zorder=2)
    plt.grid(visible=True, which="major", color="#999999", linestyle="dotted",
             alpha=0.5, zorder=0)
    plt.xlabel("Starting compound", fontweight="bold", fontsize=12)
    plt.ylabel("Correlation coefficient", fontweight="bold", fontsize=12)
    plt.title("Correlation coefficients between amino acid frequencies\n"
              r"and biosynthesis costs for $\it{Escherichia\ coli}$",
              fontsize=14, y=1.04)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(sys.argv[2], f"ecoli_cost_corrs.{ext}"),
                    bbox_inches="tight")

    plt.close()
    corr_cost_df.to_csv(os.path.join(sys.argv[2], "ecoli_cost_corrs.csv"), sep="\t")
