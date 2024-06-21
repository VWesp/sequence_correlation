import os
import sys
import numpy as np
import pandas as pd
import textwrap as tw
import collections as col
import scipy.stats as sci
from collections import Counter
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
    amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
                   "A", "G", "P", "T", "V", "L", "R", "S", "*"]

    code_ids = CodonTable.unambiguous_dna_by_id.keys()
    for code_id in code_ids:
        # get the genetic code given the ID
        codon_table = CodonTable.unambiguous_dna_by_id[code_id]
        genetic_name = " ".join(codon_table.names[:-1])
        # add the stop codons to the codon table
        for stop in codon_table.stop_codons:
            codon_table.forward_table[stop] = "*"

        # list containing the number of codons for each amino acid
        codon_count = Counter(codon_table.forward_table.values())
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

        l_index = 0
        for aa in amino_acids:
            aa_data = aa_dis_df[aa]
            codon_dis = col.defaultdict(lambda: 0)
            for codon_num in aa_data:
                codons = codon_num.split(";")
                for codon in codons:
                    c = codon.split(":")[0]
                    if(c != "XXX"):
                        n = int(codon.split(":")[1])
                        codon_dis[c] += n

            bottom = 0
            c_index = 0
            for codon,num in codon_dis.items():
                color = "grey"
                if(c_index < len(color_map)):
                    color = color_map[c_index]

                sum_aa[aa] += num
                # plot the amino acid sum for the dataset
                ax.bar(l_index-0.15, num, width=0.3, edgecolor="black",
                       linewidth=0.75, bottom=bottom, color=color, hatch="//")
                bottom += num
                c_index += 1

            l_index += 1

        sum_aa_list = list(sum_aa.values())
        x_data = np.arange(0, len(amino_acids), 1)
        # plot the theroetical amino acid mean for the genetic code
        genetic_calc_data = genetic_code_num * sum(sum_aa_list)
        ax.bar(x_data+0.15, genetic_calc_data, width=0.3, color="white",
               edgecolor="black", linewidth=0.75, hatch="\\\\")

        # value for the distance between the bar and the number line
        dif_text_max_high = max(max(sum_aa_list), max(genetic_calc_data))
        # value for the upper horizontal line
        dif_text_upper = dif_text_max_high * 0.01
        # value for the lower vertical line
        dif_text_lower = dif_text_max_high * 0.02
        # value for the text position
        dif_text_num = dif_text_max_high * 0.03
        # loop over the x-axis/bars
        for index in x_data:
            # value for the line for the current bar
            dif_text_high = max(sum_aa_list[index],
                                genetic_calc_data[index])
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
                     genetic_calc_data[index]+dif_text_upper]
            # plot the number line above the bar
            plt.plot(bar_x, bar_y, "k-", lw=1)
            # difference between the frequency of the data and the genetic code
            data_dif = sum_aa_list[index] - genetic_calc_data[index]
            # average of the frequency of the data and the genetic code
            data_ave = (sum_aa_list[index] + genetic_calc_data[index]) / 2
            # percentage difference between the frequency of the data and the
            # genetic code
            data_dif_perc = data_dif / data_ave * 100
            # plot the numnber above the number line
            plt.text((left_text+right_text)/2, dif_text_high+dif_text_num,
                     "{:.1f}".format(data_dif_perc), size=4, ha="center",
                     va="bottom")


        # calculation of Pearson correlation with and without Stop
        pcc_w_stop,pcc_p_w_stop = sci.pearsonr(sum_aa_list, genetic_code_num)
        ppc_text = "pcc+: {:.2f}; $p_{{pcc}}$+: {:.1e}".format(pcc_w_stop,
                                                               pcc_p_w_stop)
        pcc_wo_stop,pcc_p_wo_stop = sci.pearsonr(sum_aa_list[:-1],
                                                 genetic_code_num[:-1])
        ppc_wo_text = "pcc-: {:.2f}; $p_{{pcc}}$-: {:.1e}".format(pcc_wo_stop,
                                                                  pcc_p_wo_stop)
        ax.text(1.02, 0.98, ppc_text+"\n"+ppc_wo_text, transform=ax.transAxes,
                fontsize=8, verticalalignment="top", bbox=dict(boxstyle="round",
                facecolor="white", edgecolor="grey", alpha=0.5))


        # calculation of Spearman correlation with and without Stop
        r2_w_stop,r2_p_w_stop = sci.spearmanr(sum_aa_list, genetic_code_num)
        r2_text = "$r_2$+: {:.2f}; $p_{{r2}}$+: {:.1e}".format(r2_w_stop,
                                                               r2_p_w_stop)
        r2_wo_stop,r2_p_wo_stop = sci.spearmanr(sum_aa_list[:-1],
                                                genetic_code_num[:-1])
        r2_wo_text = "$r_2$-: {:.2f}; $p_{{r2}}$-: {:.1e}".format(r2_wo_stop,
                                                                  r2_p_wo_stop)
        ax.text(1.02, 0.88, r2_text+"\n"+r2_wo_text, transform=ax.transAxes,
                fontsize=8, verticalalignment="top", bbox=dict(boxstyle="round",
                facecolor="white", edgecolor="grey", alpha=0.5))

        labels = [""]*(max_codon_num*2-2) + ["Genomic data", "Genetic code"]
        plt.legend(handles=patches, labels=labels, ncol=max_codon_num,
                   handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
                   loc="best")
        # x label
        plt.xlabel("Amino acids (plus Stop)")
        # y label
        plt.ylabel("Summed amino acid number")
        # set the x-axis ticks to the amino acids
        plt.xticks(np.arange(len(amino_acids)), amino_acids)

        title_text = "Correlation between amino acid and codon number for genetic code:"
        title = "{}, {}\n{}".format(plot_name, title_text, genetic_name)
        title = "\n".join(tw.wrap(title, 50))
        plt.title(title)

        # output
        plot_output = os.path.join(output,
                                   "_".join(genetic_name.lower().split(" ")))
        plt.savefig(plot_output+".svg", bbox_inches="tight")
        plt.savefig(plot_output+".pdf", bbox_inches="tight")
        plt.close()
