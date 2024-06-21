import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as sci
from Bio.Data import CodonTable
from collections import Counter


# method for creating the codon table with the stop codons and a dictionary
# containing the number of codons for each amino acid codons given the genetic
# code ID and the order of amino acids
def createCodonTable(code_id, amino_acids):
    # get the genetic code given the ID
    codon_table = CodonTable.unambiguous_dna_by_id[code_id]
    # add the stop codons to the codon table
    for stop in codon_table.stop_codons:
        codon_table.forward_table[stop] = "*"

    genetic_name = " ".join(codon_table.names[:-1])
    # dictionary containing the number of codons for each amino acid
    codon_num = Counter(codon_table.forward_table.values())
    genetic_code_num = {aa:codon_num[aa]/np.sum(list(codon_num.values()))
                                                          for aa in amino_acids}

    # return the number of codons for the genetic code and its name
    return [genetic_code_num, " ".join(codon_table.names[:-1])]


# main method
if __name__ == "__main__":
    # path to the folder with the CSV files
    path_to_folder = sys.argv[1]
    # path to the output file
    output = sys.argv[2]

    os.makedirs(output, exist_ok=True)

    # list of amino acids
    amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
                   "A", "G", "P", "T", "V", "L", "R", "S", "*"]

    csv_files = [os.path.join(path_to_folder, file)
                 for file in os.listdir(path_to_folder)
                 if file.endswith("_aa_distribution.csv")]

    code_ids = CodonTable.unambiguous_dna_by_id.keys()
    data_list = []
    progress = 0
    prog_len = len(csv_files) * len(code_ids) * 2
    print("Plot progress: {:.2f}%".format(progress/prog_len*100), end="")
    for file in csv_files:
        row_list =[]
        prot_id = os.path.basename(file).split("_")[0]
        #tax_id = os.path.basename(file).split("_")[1]
        row_list.append(prot_id)
        df = pd.read_csv(file, sep="\t", index_col=0)
        for aa in amino_acids:
            if(not aa in df.columns):
                df[aa] = [0] * len(df)

        ordered_df = df[amino_acids]
        aa_sum = np.asarray(ordered_df.sum(axis=0))
        row_list.extend(aa_sum)
        row_list.append(np.sum(aa_sum))
        for code_id in code_ids:
            genetic_freq, genetic_name = createCodonTable(code_id, amino_acids)
            genetic_freq = np.asarray(list(genetic_freq.values()))
            pearson_w_stop, p_w_stop = sci.pearsonr(aa_sum, genetic_freq)
            row_list.append(pearson_w_stop)
            row_list.append(p_w_stop)

            pearson_wo_stop, p_wo_stop = sci.pearsonr(aa_sum[:-1],
                                                              genetic_freq[:-1])
            row_list.append(pearson_wo_stop)
            row_list.append(p_wo_stop)
            progress += 1
            print("\rPlot progress: {:.2f}%".format(progress/prog_len*100),
                                                                        end="")

        aa_mean = np.asarray(ordered_df.mean(axis=0))
        row_list.extend(aa_mean)
        row_list.append(np.mean(aa_mean))
        for code_id in code_ids:
            genetic_freq, genetic_name = createCodonTable(code_id, amino_acids)
            genetic_freq = np.asarray(list(genetic_freq.values()))
            pearson_w_stop, p_w_stop = sci.pearsonr(aa_mean, genetic_freq)
            row_list.append(pearson_w_stop)
            row_list.append(p_w_stop)

            pearson_wo_stop, p_wo_stop = sci.pearsonr(aa_mean[:-1],
                                                              genetic_freq[:-1])
            row_list.append(pearson_wo_stop)
            row_list.append(p_wo_stop)
            progress += 1
            print("\rPlot progress: {:.2f}%".format(progress/prog_len*100),
                                                                         end="")

        data_list.append(row_list)

    data_columns = ["PanProteomeID"]
    for type in ["Sum", "Mean"]:
        for aa in amino_acids:
            data_columns.append(type+" "+aa)

        data_columns.append(type+" Proteome")
        for code_id in code_ids:
            genetic_freq, genetic_name = createCodonTable(code_id, amino_acids)
            data_columns.append(type+" Pearson w/ Stop "+genetic_name)
            data_columns.append(type+" P-value w/ Stop "+genetic_name)
            data_columns.append(type+" Pearson w/o Stop "+genetic_name)
            data_columns.append(type+" P-value w/o Stop "+genetic_name)

    print()
    joined_df = pd.DataFrame(data_list, columns=data_columns)
    joined_df = joined_df.set_index("PanProteomeID")

    all_sum_mean_list = []
    for type in ["Sum", "Mean"]:
        aa_columns = [type+" "+aa for aa in amino_acids]
        aa_sum_mean_data = None
        if(type == "Sum"):
            aa_sum_mean_data = np.asarray(joined_df[aa_columns].sum(axis=0))
            all_sum_mean_list.extend(aa_sum_mean_data)
            all_sum_mean_list.append(np.sum(aa_sum_mean_data))
        elif(type == "Mean"):
            aa_sum_mean_data = np.asarray(joined_df[aa_columns].mean(axis=0))
            all_sum_mean_list.extend(aa_sum_mean_data)
            all_sum_mean_list.append(np.sum(aa_sum_mean_data))

        pearsons = []
        p_values = []
        for code_id in code_ids:
            genetic_freq, genetic_name = createCodonTable(code_id, amino_acids)
            genetic_freq = np.asarray(list(genetic_freq.values()))

            pearson_w_stop, p_w_stop = sci.pearsonr(aa_sum_mean_data,
                                                                   genetic_freq)
            all_sum_mean_list.append(pearson_w_stop)
            all_sum_mean_list.append(p_w_stop)

            pearson_wo_stop, p_wo_stop = sci.pearsonr(aa_sum_mean_data[:-1],
                                                              genetic_freq[:-1])
            all_sum_mean_list.append(pearson_wo_stop)
            all_sum_mean_list.append(p_wo_stop)

    joined_df.loc["Sum/Mean all"] = all_sum_mean_list
    joined_df.to_csv(os.path.join(output, "aa_distribution.csv"), sep="\t")
