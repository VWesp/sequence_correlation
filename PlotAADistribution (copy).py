import os
import sys
import math
import numpy as np
import sympy as sp
import pandas as pd
import seaborn as sns
import textwrap as tw
import collections as col
import scipy.stats as sci
from collections import Counter
import equation_functions as ef
from Bio.Data import CodonTable
import matplotlib.pyplot as plt
import matplotlib.patches as patch


if __name__ == "__main__":
    path_to_file = sys.argv[1]
    output = sys.argv[2]
    plot_name = sys.argv[3]

    os.makedirs(output, exist_ok=True)

    aa_dis_df = pd.read_csv(path_to_file, sep="\t", index_col=0, dtype=str)

    # list of amino acids
    amino_acids = None
    for code_id in CodonTable.unambiguous_dna_by_id.keys():
        # get the genetic code given the ID
        codon_table = CodonTable.unambiguous_dna_by_id[code_id]
        genetic_name = " ".join(codon_table.names[:-1])
        code_output = os.path.join(output,
                                   "_".join(genetic_name.lower().split(" ")))
        os.makedirs(code_output, exist_ok=True)

        aa_to_codon = col.defaultdict(lambda: [])
        for codon,aa in codon_table.forward_table.items():
            aa_to_codon[aa].append(codon)

        # list containing the number of codons for each amino acid
        codon_count = Counter(codon_table.forward_table.values())
        if(amino_acids == None):
            amino_acids = dict(sorted(codon_count.items(), key=lambda x: (x[1],
                                                                         x[0])))
            amino_acids = list(amino_acids.keys())

        genetic_code_num = np.asarray([codon_count[aa]/np.sum(list(codon_count.values()))
                                       for aa in amino_acids])

        fig, ax = plt.subplots()
        sum_aa = col.defaultdict(lambda: 0)
        max_codon_num = max(codon_count.values())
        color_space = plt.cm.rainbow(np.linspace(0, 1, max_codon_num))
        color_map = [c for i, c in enumerate(color_space)]
        patches = []
        for i in range(len(color_map)):
            patches.append(patch.Patch(facecolor=color_map[i],
                                       edgecolor="black", hatch="//"))
            patches.append(patch.Patch(facecolor="white", edgecolor="black",
                                       hatch="\\\\"))

        patches.append(patch.Patch(facecolor="grey", edgecolor="black",
                                   hatch="//"))
        patches.append(patch.Patch(facecolor="white", edgecolor="black",
                                   hatch="\\\\"))


        aa_count_lst = []
        g = sp.symbols("g", float=True)
        funcs = ef.build_functions(aa_to_codon)["amino"]
        proteome_ids = set(aa_dis_df["Genome_Tax_ID"])
        for id in proteome_ids:
            prot_aa_count = aa_dis_df[aa_dis_df["Genome_Tax_ID"]==id]
            aa_count_dic = {}
            for aa in amino_acids:
                aa_data = prot_aa_count[aa]
                aa_count = []
                for codon_num in aa_data:
                    codons = codon_num.split(";")
                    count = 0
                    for codon in codons:
                        c = codon.split(":")
                        if(c[0] != "XXX"):
                            count += int(c[1])

                    aa_count.append(count)

                aa_count_dic[aa] = np.median(aa_count)

            aa_sum = sum(list(aa_count_dic.values()))
            aa_count_dic["GC_ave"] = np.average(prot_aa_count["GC"].astype(float))
            for aa in amino_acids:
                aa_exp = aa + "_expected"
                aa_res = aa + "_residual"
                aa_count_dic[aa] = np.log2(aa_count_dic[aa]+1)
                aa_count_dic[aa_exp] = np.log2(float(funcs[aa].subs(g, aa_count_dic["GC_ave"]))*aa_sum)
                aa_count_dic[aa_res] = aa_count_dic[aa] - aa_count_dic[aa_exp]

            obs_aa_count = [aa_count_dic[aa] for aa in amino_acids]
            exp_aa_count = [aa_count_dic[aa+"_expected"] for aa in amino_acids]
            pcc,pcc_p = sci.pearsonr(obs_aa_count, exp_aa_count)
            aa_count_dic["Pearson"] = pcc
            aa_count_dic["p-value: Pearson"] = pcc_p
            r2,r2_p = sci.spearmanr(obs_aa_count, exp_aa_count)
            aa_count_dic["Spearman"] = r2
            aa_count_dic["p-value: Spearman"] = r2_p
            aa_count_lst.append(aa_count_dic)





        l_index = 0
        for aa in amino_acids:
            aa_data = aa_dis_df[aa]
            codon_dis = col.defaultdict(lambda: 0)
            for codon_num in aa_data:
                codons = codon_num.split(";")
                for codon in codons:
                    c = codon.split(":")[0]
                    if(c != "XXX"):
                        codon_dis[c] += int(codon.split(":")[1])

            bottom = 0
            c_index = 0
            for codon in aa_to_codon[aa]:
                cod_num = codon_dis[codon]
                sum_aa[aa] += cod_num
                color = color_map[c_index]
                # plot the amino acid sum for the dataset
                ax.bar(l_index-0.15, cod_num, width=0.3, edgecolor="black",
                       linewidth=0.75, bottom=bottom, color=color, hatch="//")
                bottom += cod_num
                c_index += 1

            for codon,num in codon_dis.items():
                if(not codon in aa_to_codon[aa]):
                    sum_aa[aa] += num
                    # plot the amino acid sum for the dataset
                    ax.bar(l_index-0.15, num, width=0.3, edgecolor="black",
                           linewidth=0.75, bottom=bottom, color="grey",
                           hatch="//")
                    bottom += num

            l_index += 1

        sum_aa_list = list(sum_aa.values())
        x_data = np.arange(0, len(amino_acids), 1)
        # plot the theroetical amino acid mean for the genetic code
        genetic_calc_list = genetic_code_num * sum(sum_aa_list)
        ax.bar(x_data+0.15, genetic_calc_list, width=0.3, color="white",
               edgecolor="black", linewidth=0.75, hatch="\\\\")

        # value for the distance between the bar and the number line
        dif_text_max_high = max(max(sum_aa_list), max(genetic_calc_list))
        # value for the upper horizontal line
        dif_text_upper = dif_text_max_high * 0.01
        # value for the lower vertical line
        dif_text_lower = dif_text_max_high * 0.02
        # value for the text position
        dif_text_num = dif_text_max_high * 0.03
        # loop over the x-axis/bars
        for index in x_data:
            # value for the line for the current bar
            dif_text_high = max(sum_aa_list[index], genetic_calc_list[index])
            # value for the left border
            left_text = x_data[index] - 0.15
            # value for the right border
            right_text = x_data[index] + 0.15
            # x-axis values for the line
            bar_x = [left_text-0.15, left_text-0.15, right_text+0.15,
                     right_text+0.15]
            # y-axis values for the line
            bar_y = [sum_aa_list[index]+dif_text_upper,
                     dif_text_high+dif_text_lower, dif_text_high+dif_text_lower,
                     genetic_calc_list[index]+dif_text_upper]
            # plot the number line above the bar
            plt.plot(bar_x, bar_y, "k-", lw=1)
            # difference between the frequency of the data and the genetic code
            data_dif = sum_aa_list[index] - genetic_calc_list[index]
            # average of the frequency of the data and the genetic code
            data_ave = (sum_aa_list[index] + genetic_calc_list[index]) / 2
            # percentage difference between the frequency of the data and the
            # genetic code
            data_dif_perc = data_dif / data_ave * 100
            # plot the numnber above the number line
            plt.text((left_text+right_text)/2, dif_text_high+dif_text_num,
                     "{:.1f}".format(data_dif_perc), size=4, ha="center",
                     va="bottom")


        # calculation of Pearson and Spearman correlation
        pcc,pcc_p = sci.pearsonr(sum_aa_list, genetic_code_num)
        ppc_text = "pcc: {:.2f}; $p_{{pcc}}$: {:.1e}".format(pcc, pcc_p)
        r2,r2_p = sci.spearmanr(sum_aa_list, genetic_code_num)
        r2_text = "$r_2$: {:.2f}; $p_{{r2}}$: {:.1e}".format(r2, r2_p)
        ax.text(1.02, 0.98, ppc_text+"\n"+r2_text, transform=ax.transAxes,
                fontsize=8, verticalalignment="top", bbox=dict(boxstyle="round",
                facecolor="white", edgecolor="grey", alpha=0.5))
        labels = [""]*(max_codon_num*2) + ["Genomic data", "Genetic code"]
        plt.legend(handles=patches, labels=labels, ncol=max_codon_num+1,
                   handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
                   loc="upper left")
        # x label
        plt.xlabel("Amino acid")
        # y label
        plt.ylabel("Summed amino acid number")
        # set the x-axis ticks to the amino acids
        plt.xticks(np.arange(len(amino_acids)), amino_acids)

        title_text = "Correlation between amino acid and codon number for genetic code:"
        title = "{} - {}\n{}".format(plot_name, title_text, genetic_name)
        title = "\n".join(tw.wrap(title, 50))
        plt.title(title)

        # bar output
        bar_output = os.path.join(code_output, "barplot")
        plt.savefig(bar_output+".svg", bbox_inches="tight")
        plt.savefig(bar_output+".pdf", bbox_inches="tight")
        plt.close()

        aa_count_lst = []
        g = sp.symbols("g", float=True)
        funcs = ef.build_functions(aa_to_codon)["amino"]
        proteome_ids = set(aa_dis_df["Genome_Tax_ID"])
        for id in proteome_ids:
            prot_aa_count = aa_dis_df[aa_dis_df["Genome_Tax_ID"]==id]
            aa_count_dic = {}
            for aa in amino_acids:
                aa_data = prot_aa_count[aa]
                aa_count = []
                for codon_num in aa_data:
                    codons = codon_num.split(";")
                    count = 0
                    for codon in codons:
                        c = codon.split(":")
                        if(c[0] != "XXX"):
                            count += int(c[1])

                    aa_count.append(count)

                aa_count_dic[aa] = np.median(aa_count)

            aa_sum = sum(list(aa_count_dic.values()))
            aa_count_dic["GC_ave"] = np.average(prot_aa_count["GC"].astype(float))
            for aa in amino_acids:
                aa_exp = aa + "_expected"
                aa_res = aa + "_residual"
                aa_count_dic[aa] = np.log2(aa_count_dic[aa]+1)
                aa_count_dic[aa_exp] = np.log2(float(funcs[aa].subs(g, aa_count_dic["GC_ave"]))*aa_sum)
                aa_count_dic[aa_res] = aa_count_dic[aa] - aa_count_dic[aa_exp]

            obs_aa_count = [aa_count_dic[aa] for aa in amino_acids]
            exp_aa_count = [aa_count_dic[aa+"_expected"] for aa in amino_acids]
            pcc,pcc_p = sci.pearsonr(obs_aa_count, exp_aa_count)
            aa_count_dic["Pearson"] = pcc
            aa_count_dic["p-value: Pearson"] = pcc_p
            r2,r2_p = sci.spearmanr(obs_aa_count, exp_aa_count)
            aa_count_dic["Spearman"] = r2
            aa_count_dic["p-value: Spearman"] = r2_p
            aa_count_lst.append(aa_count_dic)

        exp_amino_acids = [aa+"_expected" for aa in amino_acids]
        res_amino_acids = [aa+"_residual" for aa in amino_acids]
        aa_proteome_df = pd.DataFrame(aa_count_lst, columns=(["GC_ave"]+
                                      amino_acids+exp_amino_acids+
                                      res_amino_acids+["Pearson",
                                      "p-value: Pearson", "Spearman",
                                      "p-value: Spearman"]))
        aa_proteome_df = aa_proteome_df.sort_values("GC_ave", ascending=False)
        # plotting the data heatmap
        title = plot_name + " - {} amino acid count for genetic code: {}"
        fig, ax = plt.subplots(figsize=(10, 6))
        yticks = np.linspace(0, len(aa_proteome_df["GC_ave"])-1, 20, dtype=int)
        yticklabels = ["{:.2f}".format(list(aa_proteome_df["GC_ave"])[idx])
                       for idx in yticks]
        ax = sns.heatmap(aa_proteome_df[amino_acids], cmap="YlGnBu",
                         cbar_kws={"label": "log2-Median amino acid count"})
        b,t = ax.get_ylim()
        for i in range(1, len(amino_acids), 1):
            ax.vlines(x=i, ymin=b, ymax=t, colors="white", lw=1)

        ax.set_yticks(yticks, yticklabels)
        obs_title = "\n".join(tw.wrap(title.format("Observed", genetic_name),
                              80))
        ax.set_title(obs_title)
        ax.set_xlabel("Amino acid")
        ax.set_ylabel("GC content")

        # observed heatmap output
        heatmap_output = os.path.join(code_output, "obs_heatmap")
        plt.savefig(heatmap_output+".svg", bbox_inches="tight")
        plt.savefig(heatmap_output+".pdf", bbox_inches="tight")
        plt.close()


        # plotting the expected heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = sns.heatmap(aa_proteome_df[exp_amino_acids], cmap="YlGnBu",
                         cbar_kws={"label": "log2-Median amino acid count"})
        b,t = ax.get_ylim()
        for i in range(1, len(amino_acids), 1):
            ax.vlines(x=i, ymin=b, ymax=t, colors="white", lw=1)

        ax.set_xticks(np.arange(0.5, len(amino_acids), 1), amino_acids,
                      rotation=0)
        ax.set_yticks(yticks, yticklabels)
        exp_title = "\n".join(tw.wrap(title.format("Expected", genetic_name),
                              80))
        ax.set_title(exp_title)
        ax.set_xlabel("Amino acid")
        ax.set_ylabel("GC content")

        # expected heatmap output
        heatmap_output = os.path.join(code_output, "exp_heatmap")
        plt.savefig(heatmap_output+".svg", bbox_inches="tight")
        plt.savefig(heatmap_output+".pdf", bbox_inches="tight")
        plt.close()

        # plotting the residual heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = sns.heatmap(aa_proteome_df[res_amino_acids], cmap="coolwarm",
                         cbar_kws={"label": "Residual (Observed - Expected)"})
        b,t = ax.get_ylim()
        for i in range(1, len(amino_acids), 1):
            ax.vlines(x=i, ymin=b, ymax=t, colors="white", lw=1)

        ax.set_xticks(np.arange(0.5, len(amino_acids), 1), amino_acids,
                      rotation=0)
        ax.set_yticks(yticks, yticklabels)
        res_title = "{} - Residual between observed and expected for genetic code {}: "
        res_title = "\n".join(tw.wrap(title.format(plot_name, genetic_name),
                              80))
        ax.set_title(res_title)
        ax.set_xlabel("Amino acid")
        ax.set_ylabel("GC content")

        # residual heatmap output
        heatmap_output = os.path.join(code_output, "res_heatmap")
        plt.savefig(heatmap_output+".svg", bbox_inches="tight")
        plt.savefig(heatmap_output+".pdf", bbox_inches="tight")
        plt.close()

        # plotting pearson/spearman correlation
        fig, ax = plt.subplots(figsize=(10, 6))
        pearson_df = aa_proteome_df[aa_proteome_df["p-value: Pearson"] < 0.05]
        ax.scatter(pearson_df["GC_ave"], pearson_df["Pearson"], alpha=0.7,
                   color="red", edgecolor="black", label="Pearson correlation")
        spearman_df = aa_proteome_df[aa_proteome_df["p-value: Spearman"] < 0.05]
        ax.scatter(spearman_df["GC_ave"], spearman_df["Spearman"], alpha=0.7,
                   color="blue", edgecolor="black", label="Spearman correlation")
        cor_title = "{} - Pearson/Spearman correlation for genetic code: {}"
        cor_title = "\n".join(tw.wrap(cor_title.format(plot_name, genetic_name),
                              80))
        ax.set_xlabel("GC content")
        ax.set_ylabel("Pearson/Spearman correlation coefficient")
        ax.set_title(cor_title)
        plt.legend(loc="upper left")

        # correlation output
        cor_output = os.path.join(code_output, "correlation")
        plt.savefig(cor_output+".svg", bbox_inches="tight")
        plt.savefig(cor_output+".pdf", bbox_inches="tight")
        plt.close()

        df_output = os.path.join(code_output, "data_heatmap.csv")
        aa_proteome_df.to_csv(df_output, sep="\t")
