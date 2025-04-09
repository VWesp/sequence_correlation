import os
import sys
import numpy as np
import pandas as pd
import collections as coll


if __name__=="__main__":
    path_to_data,output = sys.argv[1:3]
    os.makedirs(output, exist_ok=True)

    nucleotides = ["A", "C", "G", "T"]
    codons = ["".join([i,j,k]) for i in nucleotides
                               for j in nucleotides
                               for k in nucleotides]
    amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
                   "A", "G", "P", "T", "V", "L", "R", "S"]

    counts_by_aa = coll.defaultdict(lambda: coll.defaultdict(list))
    files = os.listdir(path_to_data)
    print(f"Progress: {0}/{len(files)}", end="")
    for index,file in enumerate(files):
        data = pd.read_csv(os.path.join(path_to_data, file), sep="\t",
                           header=0, index_col=0).fillna(0)
        for aa in amino_acids:
            for codon in codons:
                if(not aa in data.index or not codon in data.columns):
                    data.loc[aa, codon] = 0

        data = data.loc[amino_acids, codons]
        for aa,codon_row in data.iterrows():
            codon_row = codon_row[codon_row>0]
            expected = codon_row.sum() / len(codon_row) if len(codon_row) > 0 else 0
            for codon,count in codon_row.items():
                rscu = count / expected if expected > 0 else 0
                counts_by_aa[aa][codon].append(rscu)

        print(f"\rProgress: {index+1}/{len(files)}", end="")

    print()
    for aa,codon_rscu in counts_by_aa.items():
        for codon,rscu_list in codon_rscu.items():
            rscu_list = np.asarray(rscu_list)
            counts_by_aa[aa][codon] = np.mean(rscu_list)

    rscu_df = pd.DataFrame.from_dict(counts_by_aa)
    for aa in amino_acids:
        for codon in codons:
            if(not aa in rscu_df.columns or not codon in rscu_df.index):
                rscu_df.loc[codon, aa] = None

    rscu_df = rscu_df.loc[codons, amino_acids]
    rscu_df.to_csv(os.path.join(output, "mean_RSCUs.csv"), sep="\t")

    sns.heatmap(rscu_df, cmap="viridis")
    plt.xticks(np.arange(len(amino_acids))+0.5, amino_acids, fontsize=8,
               va="center")
    plt.yticks(np.arange(len(codons))+0.5, codons, fontsize=8, va="center")
    plt.show()
    plt.close()
