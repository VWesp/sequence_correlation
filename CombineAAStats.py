import os
import sys
import yaml
import time
import numpy as np
import sympy as sp
import pandas as pd
import scipy.stats as sci
import multiprocessing as mp
from functools import partial
import equation_functions as ef

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


# One letter code for the amino acids of the genetic codes without Stop
AA_TO_ONE_LETTER = {"Methionine": "M", "Threonine": "T", "Asparagine": "N",
                    "Lysine": "K", "Serine": "S", "Arginine": "R",
                    "Valine": "V", "Alanine": "A", "Aspartic_acid": "D",
                    "Glutamic_acid": "E", "Glycine": "G", "Phenylalanine": "F",
                    "Leucine": "L", "Tyrosine": "Y", "Cysteine": "C",
                    "Tryptophane": "W", "Proline": "P", "Histidine": "H",
                    "Glutamine": "Q", "Isoleucine": "I"}
ONE_LETTER_TO_AA = {v:k for k,v in AA_TO_ONE_LETTER.items()}


def s_corr_permut_test(x, y, permuts):
    real_corr, _ = sci.spearmanr(x, y)
    permuted_corrs = np.zeros(permuts)
    for i in range(permuts):
        permut_x = np.random.permutation(x)
        permuted_corrs[i], _ = sci.spearmanr(permut_x, y)

    real_p_value = np.mean(np.abs(permuted_corrs) >= np.abs(real_corr))
    return [real_corr, real_p_value]


def process_file(file, amino_acids, enc_df, codes, code_map_df, output, permuts,
                 prog, size, time_prog, lock):
    df = pd.read_csv(file, sep="\t", header=0, index_col=0)
    df.fillna(0.0, inplace=True)

    fold_sr = pd.Series()
    id = os.path.basename(file).split(".")[0]
    fold_sr.name = id
    fold_sr["#Proteins"] = len(df)
    for col in ["GC", "Length"]+amino_acids:
        if(col in df.columns):
            fold_sr[f"{col}_mean"] = df[col].mean()
            fold_sr[f"{col}_std"] = df[col].std()
        else:
            fold_sr[f"{col}_mean"] = 0.0
            fold_sr[f"{col}_std"] = 0.0

    tax_id = int(id.split("_")[1])
    code_id = int(enc_df.loc[tax_id, "GeneticID"])
    code_name = code_map_df.loc[code_id, "Name"]
    code_path = os.path.join(codes, code_name)
    code_df = pd.read_csv(f"{code_path}.csv", sep="\t", header=0, index_col=0)

    yaml_code = None
    with open(f"{code_path}.yaml", "r") as code_reader:
        yaml_code = yaml.safe_load(code_reader)

    # Load frequency functions for each amino acid based on the codons and
    # GC content
    freq_funcs = ef.build_functions(yaml_code)
    calc_func = ef.calculate_frequencies(freq_funcs, fold_sr["GC_mean"])
    for aa in amino_acids:
        fold_sr[f"{aa}_freq"] = calc_func["amino"][ONE_LETTER_TO_AA[aa]]

    aa_mean_cols = [f"{aa}_mean" for aa in amino_acids]
    aa_freq_cols = [f"{aa}_freq" for aa in amino_acids]
    ############################ Percentage change between empirical and code
    ############################ data/frequency data
    for aa in amino_acids:
        code_freq = code_df["Frequency"][aa]
        func_freq = fold_sr[f"{aa}_freq"]
        fold_sr[f"{aa}_pct_code"] = ((fold_sr[f"{aa}_mean"] - code_freq) / code_freq) * 100
        fold_sr[f"{aa}_pct_freq"] = ((fold_sr[f"{aa}_mean"] - func_freq) / func_freq) * 100

    ############################ Spearman code
    corr, p_corr = s_corr_permut_test(fold_sr[aa_mean_cols],
                                      code_df["Frequency"][amino_acids], permuts)
    fold_sr["Spearman_code"] = corr
    fold_sr["Spearman_code_p"] = p_corr
    ############################ Spearman frequency
    corr, p_corr = s_corr_permut_test(fold_sr[aa_mean_cols],
                                      fold_sr[aa_freq_cols], permuts)
    fold_sr["Spearman_freq"] = corr
    fold_sr["Spearman_freq_p"] = p_corr
    ############################ Kendall tau code
    corr, p_corr = sci.kendalltau(fold_sr[aa_mean_cols],
                                  code_df["Frequency"][amino_acids],
                                  nan_policy="raise")
    fold_sr["Kendall_code"] = corr
    fold_sr["Kendall_code_p"] = p_corr
    ############################ Kendall tau frequency
    corr, p_corr = sci.kendalltau(fold_sr[aa_mean_cols], fold_sr[aa_freq_cols],
                                  nan_policy="raise")
    fold_sr["Kendall_freq"] = corr
    fold_sr["Kendall_freq_p"] = p_corr

    fold_sr["Genetic_code"] = code_name

    with lock:
        prog.value += 1
        elapsed_time = time.strftime("%Hh:%Mm:%Ss", time.gmtime(time.time()-time_prog.value))
        print(f"\rFiles: {int(prog.value)}/{size} -> {(prog.value/size)*100:.2f}% -> Elapsed time: {elapsed_time}",
              end="")

    return fold_sr.to_frame().T


# main method
if __name__ == "__main__":
    mp.freeze_support()
    manager = mp.Manager()
    lock = manager.Lock()
    prog = manager.Value("i", 0)
    time_prog = manager.Value("d", 0)

    path_to_data,output,encoding,codes,code_map,permuts,procs = sys.argv[1:8]
    os.makedirs(os.path.dirname(output), exist_ok=True)

    code_map_df = pd.read_csv(code_map, sep="\t", header=0, index_col=0)
    enc_df = pd.read_csv(encoding, sep="\t", header=0, index_col=0)

    # Canonical amino acids order
    amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
                   "A", "G", "P", "T", "V", "L", "R", "S"]

    abund_files = [os.path.join(path_to_data, file)
                   for file in os.listdir(path_to_data)]

    size = len(abund_files)
    time_prog.value = time.time()
    elapsed_time = time.strftime("%Hh:%Mm:%Ss", time.gmtime(time.time()-time_prog.value))
    print(f"Files: {int(prog.value)}/{size} -> {prog.value:.2f}% -> Elapsed time: {elapsed_time}", end="")
    result = None
    with mp.Pool(processes=int(procs)) as pool:
        # run the process for the given parameters
        pool_map = partial(process_file, amino_acids=amino_acids, enc_df=enc_df,
                           codes=codes, code_map_df=code_map_df, output=output,
                           permuts=int(permuts), prog=prog, size=size,
                           time_prog=time_prog, lock=lock)
        result = pool.map_async(pool_map, abund_files)
        pool.close()
        pool.join()

        comb_df = pd.DataFrame()
        for res in result.get():
            comb_df = pd.concat([comb_df, res])

        comb_df.fillna(0.0, inplace=True)
        comb_df.index.name = "Prot_Tax_ID"
        comb_df.to_csv(output, sep="\t")

    print()
