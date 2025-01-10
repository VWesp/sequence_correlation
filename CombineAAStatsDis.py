import os
import sys
import yaml
import numpy as np
import sympy as sp
import pandas as pd
import scipy.stats as sci
import collections as defcol
import sklearn.metrics as skl
import equation_functions as ef


# One letter code for the amino acids of the genetic codes
AA_TO_ONE_LETTER = {"Methionine": "M", "Threonine": "T", "Asparagine": "N",
                    "Lysine": "K", "Serine": "S", "Arginine": "R",
                    "Valine": "V", "Alanine": "A", "Aspartic_acid": "D",
                    "Glutamic_acid": "E", "Glycine": "G", "Phenylalanine": "F",
                    "Leucine": "L", "Tyrosine": "Y", "Cysteine": "C",
                    "Tryptophane": "W", "Proline": "P", "Histidine": "H",
                    "Glutamine": "Q", "Isoleucine": "I", "Stop": "*"}


def combine_data(paths, amino_acids):
    comb_df = pd.DataFrame()
    for prot in prot_dir:
        df = pd.read_csv(prot, sep="\t", header=0, index_col=0)
        df = df.drop(["Status"], axis=1)
        df_mean = df.mean(axis=0)
        df_mean["#Proteins"] = len(df)
        df_mean.name = os.path.basename(prot).split("_")[0]
        comb_df = pd.concat([comb_df, df_mean.to_frame().T])

    comb_df = comb_df.fillna(0.0)
    if("X" in comb_df.columns):
        comb_df.loc[:, amino_acids] = comb_df.loc[:, amino_acids].add(comb_df["X"]/20, axis=0)
        comb_df = comb_df.drop(["X"], axis=1)

    if("B" in comb_df.columns):
        comb_df.loc[:, ["N", "D"]] = comb_df.loc[:, ["N", "D"]].add(comb_df["B"]/2, axis=0)
        comb_df = comb_df.drop(["B"], axis=1)

    if("Z" in comb_df.columns):
        comb_df.loc[:, ["Q", "E"]] = comb_df.loc[:, ["Q", "E"]].add(comb_df["Z"]/2, axis=0)
        comb_df = comb_df.drop(["Z"], axis=1)

    if("O" in comb_df.columns):
        comb_df.loc[:, ["K"]] = comb_df.loc[:, ["K"]].add(comb_df["O"], axis=0)
        comb_df = comb_df.drop(["O"], axis=1)

    if("U" in comb_df.columns):
        comb_df.loc[:, ["C"]] = comb_df.loc[:, ["C"]].add(comb_df["U"], axis=0)
        comb_df = comb_df.drop(["U"], axis=1)

    comb_df.index.name = "Proteome_ID"
    comb_df = comb_df[["GC", "Length", "#Proteins"]+amino_acids]
    return comb_df


if __name__ == "__main__":
    path,codes,type,output = sys.argv[1:5]

    # Canonical amino acids
    amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
                   "A", "G", "P", "T", "V", "L", "R", "S"]

    kingdoms = ["Archaea", "Bacteria", "Eukaryota", "Viruses"]
    king_freq_data = defcol.defaultdict(lambda: defcol.defaultdict(lambda:{}))
    standard_norm_data = None
    standard_freq_data = {}

    for kingdom in kingdoms:
        print(f"Current kingdom: {kingdom}...")
        prot_input = os.path.join(os.path.join(path, kingdom.lower()), os.path.join("aa_count_results", type))
        prot_output = os.path.join(os.path.join(output, type), kingdom.lower())
        os.makedirs(prot_output, exist_ok=True)

        prot_dir = [os.path.join(prot_input, file) for file in os.listdir(prot_input)]
        code_dir = [os.path.join(codes, file) for file in os.listdir(codes)]

        prot_df = combine_data(prot_dir, amino_acids)
        prot_df[amino_acids] = prot_df[amino_acids].div(prot_df[amino_acids].sum(axis=1),
                                                                                 axis=0)
        entropies = []
        for i,row in prot_df.iterrows():
            entropy = -sum([row[aa] * np.log2(row[aa]) if row[aa] > 0 else 0
                            for aa in amino_acids])
            entropies.append(entropy)

        prot_df["shannon_entropy"] = entropies
        prot_df.to_csv(os.path.join(prot_output, "proteome_data.csv"), sep="\t")

        for code in code_dir:
            code_name = os.path.basename(code).split(".yaml")[0]
            print(f"Current code: {code_name}")
            code_output = os.path.join(prot_output, code_name)
            os.makedirs(code_output, exist_ok=True)

            # Load genetic code file
            yaml_code = None
            with open(code, "r") as code_reader:
                yaml_code = yaml.safe_load(code_reader)

            # Map amino acids to their one letter code
            codon_map = {AA_TO_ONE_LETTER[aa]:codons
                         for aa,codons in yaml_code.items()}
            codon_map = {aa:codon_map[aa] for aa in amino_acids}

            # Calculate the frequency percentage for each amino acid based on the
            # codon number
            total_codon_num = np.sum([len(codon_map[aa]) for aa in amino_acids])
            codon_num_norm = {aa:len(codon_map[aa])/total_codon_num
                              for aa in amino_acids}

            code_df = pd.DataFrame.from_dict(codon_num_norm, orient="index")
            code_df.to_csv(os.path.join(code_output, "norm_code_data.csv"),
                           sep="\t", header=["frequency"])

            freq_df = pd.DataFrame()
            corr_df = pd.DataFrame()

            #############################################
            # Load frequency functions for each amino acid based on the codons and
            # GC content
            g = sp.symbols("g", float=True)
            freq_funcs = ef.build_functions(codon_map)

            for i,row in prot_df.iterrows():
                row_data = np.array(row[amino_acids])
                df_data = np.array(code_df[0])
                log_data = np.log2(row_data, where=row_data!=0)

                freqs = np.array(list(ef.calculate_frequencies(freq_funcs,
                                                      row["GC"])["amino"].values()))
                freqs = freqs / freqs.sum()
                log_freqs = np.log2(freqs, where=freqs!=0)
                freq_df.loc[row.name, amino_acids] = freqs
                ##############
                corr, p_corr = sci.pearsonr(row_data, df_data)
                corr_df.loc[row.name, "codon_pear"] = corr
                corr_df.loc[row.name, "p_codon_pear"] = p_corr
                ##############
                corr, p_corr = sci.spearmanr(row_data, df_data)
                corr_df.loc[row.name, "codon_spear"] = corr
                corr_df.loc[row.name, "p_codon_spear"] = p_corr
                ##############
                mse = skl.mean_squared_error(log_data, np.log2(code_df))
                corr_df.loc[row.name, "codon_log_mse"] = mse
                ############################
                corr, p_corr = sci.pearsonr(row_data, freqs)
                corr_df.loc[row.name, "gc_pear"] = corr
                corr_df.loc[row.name, "p_gc_pear"] = p_corr
                ##############
                corr, p_corr = sci.spearmanr(row_data, freqs)
                corr_df.loc[row.name, "gc_spear"] = corr
                corr_df.loc[row.name, "p_gc_spear"] = p_corr
                ##############
                mse = skl.mean_squared_error(log_data, log_freqs)
                corr_df.loc[row.name, "gc_log_mse"] = mse
                ############################
                entropy = -sum([freqs[i] * np.log2(freqs[i]) if freqs[i] > 0 else 0
                                for i in range(len(freqs))])
                corr_df.loc[row.name, "shannon_entropy"] = entropy

            freq_df.index.name = "Proteome_ID"
            freq_df.to_csv(os.path.join(code_output, "pred_freq_data.csv"), sep="\t")

            corr_df.index.name = "Proteome_ID"
            corr_df.to_csv(os.path.join(code_output, "corr_data.csv"), sep="\t")

        print()
