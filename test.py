import yaml
import numpy as np
import sympy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import equation_functions as ef
from settings import AA_TO_ONE_LETTER


amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
               "A", "G", "P", "T", "V", "L", "R", "S"]
aa_test = [f"{aa}_test" for aa in amino_acids]
# Load genetic code file
yaml_code = None
with open("genetic_codes/standard.yaml", "r") as code_reader:
    yaml_code = yaml.safe_load(code_reader)

# Map amino acids to their one letter code
codon_map = {AA_TO_ONE_LETTER[aa]:yaml_code[aa]
             for aa,codons in yaml_code.items()}
codon_map = {aa:codon_map[aa] for aa in amino_acids}

df = pd.read_csv("species_sprot/eukaryota/aa_code_correlations/standard/proteome_cor_data.csv", sep="\t", index_col=0)
print(df)
df = df.groupby("Genome_Tax_ID").mean()["GC"]
print(df)

list = [[1,2,3,5,5]]*3
print(list)
fdjksjfsd

g = sp.symbols("g", float=True)
freq_funcs = ef.build_functions(codon_map)
freq_list = []
for gc in df["GC"]:
    freqs = list(ef.calculate_frequencies(freq_funcs, gc)["amino"].values())
    freq_list.append(freqs)

df[aa_test] = freq_list
print(df)

'''
print(median_df[amino_acids])
aa_exp = [f"{aa}_exp" for aa in amino_acids]
median_df[aa_exp] = np.log(median_df[amino_acids])
print(median_df[aa_exp])
median_df["cost_mean"] = -np.mean(median_df[aa_exp], axis=1)
print(median_df["cost_mean"])
aa_en_cost = [f"{aa}_en_cost" for aa in amino_acids]
median_df[aa_en_cost] = median_df.apply(lambda row: -row[aa_exp]-row["cost_mean"], axis=1)
print(median_df[aa_en_cost])
aa_en_neg_cost = [f"{aa}_en_neg_cost" for aa in amino_acids]
median_df[aa_en_neg_cost] = np.exp(-median_df[aa_en_cost])
print(median_df[aa_en_neg_cost])
median_df["sum_exp"] = np.sum(median_df[aa_en_neg_cost], axis=1)
print(median_df["sum_exp"])
aa_freq_obs = [f"{aa}_freq_obs" for aa in amino_acids]
median_df[aa_freq_obs] = median_df[aa_en_neg_cost].div(median_df["sum_exp"], axis=0)
print(median_df[aa_freq_obs])
'''
