import os
import sys
import yaml
import time
import numpy as np
import sympy as sp
import pandas as pd
import scipy.stats as sci
import collections as defcol
import multiprocessing as mp
from functools import partial
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

amino_acids = AA_TO_ONE_LETTER.values()


def combine_data(paths, amino_acids):
    comb_mean_df = pd.DataFrame()
    comb_std_df = pd.DataFrame()
    for prot in prot_dir:
        df = pd.read_csv(prot, sep="\t", header=0, index_col=0)
        df = df.drop(["Status"], axis=1)

        df_mean = df.mean(axis=0)
        df_mean["#Proteins"] = len(df)
        df_mean.name = os.path.basename(prot).split("_")[0]
        comb_mean_df = pd.concat([comb_mean_df, df_mean.to_frame().T])

        df_std = df.std(axis=0)
        df_std["#Proteins"] = len(df)
        df_std.name = os.path.basename(prot).split("_")[0]
        comb_std_df = pd.concat([comb_std_df, df_std.to_frame().T])

    comb_mean_df = comb_mean_df.fillna(0.0)
    comb_std_df = comb_std_df.fillna(0.0)

    additional_cols = ["#Proteins", "GC", "Length"]
    non_canoncical_aa_mean_freqs = {aa_col:[np.mean(comb_mean_df[aa_col])]
                                    for aa_col in comb_mean_df.columns
                                    if not aa_col in amino_acids
                                       and not aa_col in additional_cols}
    non_canoncical_df = pd.DataFrame.from_dict(non_canoncical_aa_mean_freqs)

    comb_mean_df.index.name = "Proteome_ID"
    comb_mean_df = comb_mean_df[additional_cols+amino_acids]

    comb_std_df.index.name = "Proteome_ID"
    comb_std_df = comb_std_df[additional_cols+amino_acids]

    return [comb_mean_df, comb_std_df, non_canoncical_df]


def calculate_gc_freqs(gc, func):
    freqs = ef.calculate_frequencies(func, gc)["amino"]
    return pd.Series(freqs)


def corr_permut_test(x, y, permuts):
    real_corr, _ = sci.spearmanr(x, y)
    permuted_corrs = np.zeros(permuts)
    for i in range(permuts):
        permut_x = np.random.permutation(x)
        permuted_corrs[i], _ = sci.spearmanr(permut_x, y)

    real_p_value = np.mean(np.abs(permuted_corrs) >= np.abs(real_corr))
    return [real_corr, real_p_value]


def process_code(code, df, output, permuts, prog, size, time_prog, lock):
    code_name = os.path.basename(code).split(".yaml")[0]
    code_output = os.path.join(output, code_name)
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

    # Load frequency functions for each amino acid based on the codons and
    # GC content
    freq_funcs = ef.build_functions(codon_map)

    freq_df = pd.DataFrame()
    corr_code_df = pd.DataFrame()
    corr_gc_df = pd.DataFrame()
    code_data = np.array(code_df[0])
    for i,row in df.iterrows():
        row_data = np.array(row[amino_acids])
        freq_data = np.array(calculate_gc_freqs(row["GC"], freq_funcs))
        freq_df.loc[row.name, amino_acids] = freq_data
        ############################
        corr, p_corr = corr_permut_test(row_data, code_data, permuts)
        corr_code_df.loc[row.name, "spearman"] = corr
        corr_code_df.loc[row.name, "p_spearman"] = p_corr
        ##############
        corr, p_corr = sci.kendalltau(row_data, code_data, nan_policy="raise")
        corr_code_df.loc[row.name, "kendall"] = corr
        corr_code_df.loc[row.name, "p_kendall"] = p_corr
        ###################
        corr, p_corr = corr_permut_test(row_data, freq_data, permuts)
        corr_gc_df.loc[row.name, "spearman"] = corr
        corr_gc_df.loc[row.name, "p_spearman"] = p_corr
        ##############
        corr, p_corr = sci.kendalltau(row_data, freq_data, nan_policy="raise")
        corr_gc_df.loc[row.name, "kendall"] = corr
        corr_gc_df.loc[row.name, "p_kendall"] = p_corr

        with lock:
            prog.value += 1
            elapsed_time = time.strftime("%Hh:%Mm:%Ss", time.gmtime(time.time()-time_prog.value))
            print("\rKingdom: {} -> {:.2f}% -> Elapsed time: {}".format(kingdom,
                  (prog.value/size)*100, elapsed_time), end="")

    freq_df.index.name = "Proteome_ID"
    freq_df.to_csv(os.path.join(code_output, "pred_freq_data.csv"), sep="\t")

    corr_code_df.index.name = "Proteome_ID"
    corr_code_df.to_csv(os.path.join(code_output, "corr_code_data.csv"), sep="\t")

    corr_gc_df.index.name = "Proteome_ID"
    corr_gc_df.to_csv(os.path.join(code_output, "corr_gc_data.csv"), sep="\t")


if __name__ == "__main__":
    mp.freeze_support()
    manager = mp.Manager()
    lock = manager.Lock()
    time_prog = manager.Value("d", 0)

    path,seq_type,codes,permuts,output,procs = sys.argv[1:7]

    # Canonical amino acids
    amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
                   "A", "G", "P", "T", "V", "L", "R", "S"]

    kingdoms = ["Archaea", "Bacteria", "Eukaryotes", "Viruses"]
    for kingdom in kingdoms:
        prog = manager.Value("i", 0)
        time_prog.value = time.time()
        elapsed_time = time.strftime("%Hh:%Mm:%Ss", time.gmtime(time.time()-time_prog.value))
        print("Kingdom: {} -> {:.2f}% -> Elapsed time: {}".format(kingdom,
                                              prog.value, elapsed_time), end="")
        prot_input = os.path.join(os.path.join(path, kingdom.lower()),
                                  os.path.join("abundances", seq_type))
        prot_output = os.path.join(os.path.join(output, seq_type), kingdom.lower())
        os.makedirs(prot_output, exist_ok=True)

        prot_dir = [os.path.join(prot_input, file) for file in os.listdir(prot_input)]
        code_dir = [os.path.join(codes, file) for file in os.listdir(codes)]

        dfs = combine_data(prot_dir, amino_acids)
        dfs[1].to_csv(os.path.join(prot_output, "proteome_std_data.csv"),
                      sep="\t")
        dfs[2].to_csv(os.path.join(prot_output, "non_canoncical_aa_mean_freqs.csv"),
                      sep="\t")

        prot_mean_df = dfs[0]
        prot_mean_df[amino_acids] = prot_mean_df[amino_acids].div(prot_mean_df[amino_acids].sum(axis=1),
                                                                  axis=0)
        prot_mean_df.to_csv(os.path.join(prot_output, "proteome_mean_data.csv"),
                            sep="\t")

        size = len(prot_mean_df) * len(code_dir)
        # start the multicore process for a given number of threads
        with mp.Pool(processes=int(procs)) as pool:
            # run the process for the given parameters
            pool_map = partial(process_code, df=prot_mean_df, output=prot_output,
                               permuts=int(permuts), prog=prog, size=size,
                               time_prog=time_prog, lock=lock)
            process = pool.map_async(pool_map, code_dir)
            pool.close()
            #print(process.get())
            pool.join()

        elapsed_time = time.strftime("%Hh:%Mm:%Ss", time.gmtime(time.time()-time_prog.value))
        print("\rKingdom: {} -> {:.2f}% -> Elapsed time: {}".format(kingdom,
                                                             100, elapsed_time))
