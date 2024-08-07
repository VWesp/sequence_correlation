import os
import sys
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import collections as col
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def rev_con(x, list):
    for l in list:
        if l in x:
            return True

    return False


# main method
if __name__ == "__main__":
    file_path = sys.argv[1]
    output = sys.argv[2]
    plot_name= sys.argv[3]

    # Load the new file with amino acid homo repeats for eukaryotic proteins
    data = pd.read_csv(file_path, sep='\t')

    # Prepare the data for PCA
    # Drop non-numeric columns
    numeric_data = data.drop(columns=['Prot_ID', 'Genome_Tax_ID'])

    # Handle missing values (if any) by filling with zeros
    numeric_data = numeric_data.fillna(0)

    # Standardize the data before PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)

    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Genome_Tax_ID'] = data['Genome_Tax_ID']

    # Plot the PCA result
    plt.figure(figsize=(10, 7))
    for genome_tax_id in pca_df['Genome_Tax_ID'].unique():
        indices = pca_df['Genome_Tax_ID'] == genome_tax_id
        plt.scatter(pca_df.loc[indices, 'PC1'], pca_df.loc[indices, 'PC2'], label=genome_tax_id, alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Amino Acid Homo Repeats')
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), ncol=1)
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    '''amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
                   "A", "G", "P", "T", "V", "L", "R", "S"]
    aa_reps = pd.read_csv(file_path, sep="\t")
    sel_columns = aa_reps.columns[1:-2]
    aa_reps = aa_reps[sel_columns]
    sel_columns = [col for aa in amino_acids
                   for col in aa_reps.columns if aa in col]
    aa_reps = aa_reps[sel_columns]

    # Summarize the repeat counts for each amino acid
    amino_acid_sums = aa_reps.sum(axis=0)
    aa_reps_melted = aa_reps.melt(var_name="Repeat", value_name="Count")

    # Calculate descriptive statistics for the repeat counts of each amino acid
    descriptive_stats = aa_reps_melted.groupby("Repeat").agg({
        "Count": ["mean", "median", "std"]
    }).reset_index()

    print(descriptive_stats)'''

    '''aa_reps_melted["AminoAcid"] = aa_reps_melted["Repeat"].str[0]
    aa_reps_melted["Length"] = aa_reps_melted["Repeat"].apply(lambda x: len(x))
    print(aa_reps_melted)

    # Calculate descriptive statistics for the repeat counts of each amino acid
    descriptive_stats = aa_reps_melted.groupby("Repeat").agg({
        "Count": ["mean", "median", "std"]
    }).reset_index()'''

    '''#os.makedirs(output, exist_ok=True)
    amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
                   "A", "G", "P", "T", "V", "L", "R", "S"]
    aa_reps = pd.read_csv(file_path, sep="\t")
    aa_sliced = aa_reps.head(1000)
    aa_sliced.to_csv("new_example.csv", sep="\t")
    columns = aa_reps.columns[1:-2]
    aa_reps = aa_reps[columns]

    # Summing the values for each repeat length column grouped by amino acid
    summary_data = aa_reps.sum().reset_index()
    summary_data.columns = ["AminoAcid_Repeat", "Sum"]
    summary_data["logSum"] = np.log2(summary_data["Sum"])
    summary_data.loc[summary_data["logSum"]==0, "logSum"] = 1
    summary_data["AminoAcid"] = summary_data["AminoAcid_Repeat"].str[0]
    summary_data["Length"] = summary_data["AminoAcid_Repeat"].apply(lambda x: len(x))
    summary_data = summary_data[summary_data["AminoAcid"].isin(amino_acids)]
    aa_groups = {"Charged": ["R", "H", "K", "D", "E"], "Aromatic": ["F", "Y",
                 "W"], "Aliphatic": ["G", "A", "V", "L", "M", "I"],
                 "Uncharged": ["S", "T", "C", "P", "N", "Q"]}
    color_space = plt.cm.rainbow(np.linspace(0, 1, len(aa_groups.keys())))
    color_map = {list(aa_groups.keys())[i]:c for i, c in enumerate(color_space)}
    aa_colors = {aa:color_map[type] for type,aas in aa_groups.items()
                                    for aa in aas}
    sns.violinplot(x="AminoAcid", y="Sum", data=summary_data, inner=None,
                   palette=aa_colors, cut=0)
    summary_data.to_csv("example.csv", sep="\t")
    plt.ylim(0, max(summary_data["Sum"])*1.05)
    plt.title("Violin Plot of Homorepeat Lengths and Sum of Their Appearances for Each Amino Acid")
    plt.xlabel("Amino acid")
    plt.ylabel("Number of repeats")
    plt.show()
    plt.close()'''

    '''aa_data = col.defaultdict(lambda: [[], []])
    for aa_rep in aa_reps.columns:
        if(not aa_rep in ["Length", "Genome_Tax_ID"]):
            aa = list(set(aa_rep))[0]
            aa_rep_sum = np.sum(aa_reps[aa_rep])
            aa_data[aa][0].append(len(aa_rep))
            aa_data[aa][1].append(aa_rep_sum)

    print(aa_data)

    aa_groups = {"Charged": ["R", "H", "K", "D", "E"], "Aromatic": ["F", "Y",
                 "W"], "Aliphatic": ["G", "A", "V", "L", "M", "I"],
                 "Uncharged": ["S", "T", "C", "P", "N", "Q"]}
    markers = {"Charged": "o", "Aromatic": "*", "Aliphatic": "d",
               "Uncharged": "s"}

    fig, ax = plt.subplots(figsize=(17, 10))
    x_max = 0
    y_max = 0
    x_loc = 0.76
    for group,amino_acids in aa_groups.items():
        color_space = plt.cm.rainbow(np.linspace(0, 1, len(amino_acids)))
        color_map = {amino_acids[i]:c for i, c in enumerate(color_space)}
        handles = []
        for aa in amino_acids:
            aa_rep_lengths = np.asarray(aa_data[aa][0])
            aa_rep_numbers = np.asarray(aa_data[aa][1])
            log_reps = np.asarray(np.log2(aa_rep_numbers))
            handle = ax.scatter(aa_rep_lengths, log_reps, s=80,
                                color=color_map[aa], alpha=0.7, label=aa,
                                marker=markers[group], edgecolor="black")
            handles.append(handle)

            x_max = max(x_max, max(aa_rep_lengths))
            y_max = max(y_max, max(log_reps))

        legend = ax.legend(handles=handles, title=group+"\namino acids",
                           bbox_to_anchor=(x_loc, 1))
        ax.add_artist(legend)
        x_loc += 0.08

    ax.set_xlim(0, x_max*1.05)
    ax.set_ylim(0, y_max*1.05)
    ax.grid(visible=True, which="major", color="#999999",
             linestyle="dotted", alpha=0.5, zorder=0)
    ax.set_xlabel("Repeat length")
    ax.set_ylabel("log2-Number of monorepeats")
    ax.set_title(plot_name+", Number of monorepeats for given amino acid")
    # output
    plt.savefig(os.path.join(output, "monorepeats.svg"), bbox_inches="tight",
                dpi=fig.dpi)
    plt.savefig(os.path.join(output, "monorepeats.pdf"), bbox_inches="tight",
                dpi=fig.dpi)
    plt.close()'''
