import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from settings import AA_TO_ONE_LETTER

def test(stuff):
    print(stuff)

def calcCostFreq(df):
    df = df.copy()
    df.replace(0, 1, inplace=True)
    # Calculate the negative natural logarithm of each amino acid frequency
    aa_ln = np.log(df)
    # Estimate constant C
    const_c = -np.mean(aa_ln, axis=1)
    # Calculate energetic cost
    en_cost = (-aa_ln).subtract(const_c, axis=0)
    # Calculate the exponentials of the negative energetic costs
    exp_neg_cost = np.exp(-en_cost)
    # Sum the exponentials
    sum_exp = np.sum(exp_neg_cost, axis=1)
    # Return the frequency of each amino acid
    return exp_neg_cost.div(sum_exp, axis=0)

amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
               "A", "G", "P", "T", "V", "L", "R", "S"]

df = pd.read_csv("species_sprot/eukaryota/aa_code_correlations/standard/proteome_cor_data.csv", sep="\t", index_col=0)
median_df = df.groupby("Genome_Tax_ID").median()
aa_ener = [f"{aa}_ener" for aa in amino_acids]
median_df[aa_ener] = calcCostFreq(median_df[amino_acids])
#df = df.merge(median_df[aa_ener], left_on="Genome_Tax_ID", right_index=True)
df.apply(lambda row: test(row), axis=1)

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
