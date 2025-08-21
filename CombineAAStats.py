import os
import tqdm
import yaml
import argparse
import numpy as np
import pandas as pd
import scipy.stats as sci
import multiprocessing as mp
import sklearn.metrics as skl
import equation_functions as ef
from statsmodels.stats.weightstats import DescrStatsW


def fisher_Z(x):
	### Fisher's Z-transformation
	z = [0.5*np.log((1+r)/(1-r)) for r in x]
	mean_z = np.mean(z)
	std_z = np.std(z)
	mean_r = (np.exp(2*mean_z)-1) / (np.exp(2*mean_z)+1)
	std_r = (np.exp(2*std_z)-1) / (np.exp(2*std_z)+1)
	return [mean_r, std_r]

# Canonical amino acids order
amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I", "A", "G", "P", "T", "V", "L", "R", "S"]
	
# Full name for each amino acid
one_letter_code = {"M": "Methionine", "T": "Threonine", "N": "Asparagine", "K": "Lysine", "S": "Serine", "R": "Arginine", "V": "Valine", "A": "Alanine", "D": "Aspartic_acid",
				   "E": "Glutamic_acid", "G": "Glycine", "F": "Phenylalanine", "L": "Leucine", "Y": "Tyrosine", "C": "Cysteine", "W": "Tryptophane", "P": "Proline",
				   "H": "Histidine", "Q": "Glutamine", "I": "Isoleucine"}


def combine_distribution_stats(data):
	tax_id,dis_df,code_name,freq_funcs,resamples = data

	dis_df.fillna(0.0, inplace=True)
	dis_sr = pd.Series(name=tax_id)
	dis_sr["#Proteins"] = len(dis_df)
	dis_sr["Length_mean"] = dis_df["Length"].mean()
	dis_sr["Length_std"] = dis_df["Length"].std()
	dis_sr["GC_mean"] = dis_df["GC"].mean()
	dis_sr["GC_std"] = dis_df["GC"].std()
	 # Load frequency functions for each amino acid based on the codons and independent of GC content (GC=50%)
	code_freq_func = ef.calculate_frequencies(freq_funcs, 0.5)
	 # Load frequency functions for each amino acid based on the codons and GC content
	gc_freq_func = ef.calculate_frequencies(freq_funcs, dis_sr["GC_mean"])
	for aa in amino_acids:
		try:
			dis_sr[f"{aa}_mean"] = dis_df[aa].mean()
			dis_sr[f"{aa}_std"] = dis_df[aa].std()
		except KeyError:
			dis_sr[f"{aa}_mean"] = 0.0
			dis_sr[f"{aa}_std"] = 0.0
		
		obs = dis_sr[f"{aa}_mean"] if dis_sr[f"{aa}_mean"] > 0 else 1e-10
		############################ log-Fold change and percentage change between empirical and code data
		dis_sr[f"{aa}_code"] = code_freq_func["amino"][one_letter_code[aa]]
		dis_sr[f"{aa}_code_lfc"] = np.log(obs / dis_sr[f"{aa}_code"])
		dis_sr[f"{aa}_code_pct"] = (obs - dis_sr[f"{aa}_code"]) / dis_sr[f"{aa}_code"]
		############################ log-Fold change and percentage change between empirical and frequency data
		dis_sr[f"{aa}_gc"] = gc_freq_func["amino"][one_letter_code[aa]]
		dis_sr[f"{aa}_gc_lfc"] = np.log(obs / dis_sr[f"{aa}_gc"])
		dis_sr[f"{aa}_gc_pct"] = (obs - dis_sr[f"{aa}_gc"]) / dis_sr[f"{aa}_gc"]
	

	aa_mean_cols = [f"{aa}_mean" for aa in amino_acids]
	code_freq_cols = [f"{aa}_code" for aa in amino_acids]
	gc_freq_cols = [f"{aa}_gc" for aa in amino_acids]
	x_obs = np.asarray(dis_sr[aa_mean_cols], dtype=float)
	y_code = np.asarray(dis_sr[code_freq_cols], dtype=float)
	y_gc = np.asarray(dis_sr[gc_freq_cols], dtype=float)
	############################ log-RMSE
	dis_sr["log_RMSE_code"] = np.log(skl.root_mean_squared_error(x_obs, y_code))
	dis_sr["log_RMSE_gc"] = np.log(skl.root_mean_squared_error(x_obs, y_gc))
	############################ Pearson code
	corr_stats = sci.permutation_test((x_obs,), lambda x: sci.pearsonr(x, y_code).statistic, permutation_type="pairings", n_resamples=resamples)
	dis_sr["Ps_code"] = corr_stats.statistic
	dis_sr["Ps_code_p"] = corr_stats.pvalue
	############################ Pearson frequency
	corr_stats = sci.permutation_test((x_obs,), lambda x: sci.pearsonr(x, y_gc).statistic, permutation_type="pairings", n_resamples=resamples)
	dis_sr["Ps_gc"] = corr_stats.statistic
	dis_sr["Ps_gc_p"] = corr_stats.pvalue
	############################ Spearman code
	corr_stats = sci.permutation_test((x_obs,), lambda x: sci.spearmanr(x, y_code).statistic, permutation_type="pairings", n_resamples=resamples)
	dis_sr["Sm_code"] = corr_stats.statistic
	dis_sr["Sm_code_p"] = corr_stats.pvalue
	############################ Spearman frequency
	corr_stats = sci.permutation_test((x_obs,), lambda x: sci.spearmanr(x, y_gc).statistic, permutation_type="pairings", n_resamples=resamples)
	dis_sr["Sm_gc"] = corr_stats.statistic
	dis_sr["Sm_gc_p"] = corr_stats.pvalue
	############################ Kendall tau code
	corr_stats = sci.permutation_test((x_obs,), lambda x: sci.kendalltau(x, y_code).statistic, permutation_type="pairings", n_resamples=resamples)
	dis_sr["Kt_code"] = corr_stats.statistic
	dis_sr["Kt_code_p"] = corr_stats.pvalue
	############################ Kendall tau frequency
	corr_stats = sci.permutation_test((x_obs,), lambda x: sci.kendalltau(x, y_gc).statistic, permutation_type="pairings", n_resamples=resamples)
	dis_sr["Kt_gc"] = corr_stats.statistic
	dis_sr["Kt_gc_p"]  = corr_stats.pvalue
		
	dis_sr["Genetic_code"] = code_name													
	return dis_sr.to_frame().T


# main method
if __name__ == "__main__":
	mp.freeze_support()
	
	parser = argparse.ArgumentParser(description="Compute amino acid distributions in proteomes")
	parser.add_argument("-d", "--data", help="Specify the path to the folder with the distribution files", required=True)
	parser.add_argument("-o", "--output", help="Specify the output folder", required=True)
	parser.add_argument("-e", "--encoding", help="Set the path to the encoding file", required=True)
	parser.add_argument("-c", "--codes", help="Specify the path to the folder with the genetic code files", required=True)
	parser.add_argument("-m", "--mapping", help="Set the path to the mappings of the genetic codes", required=True)
	parser.add_argument("-r", "--resamples", help="Specify the number of resamples for the permutation tests (default: 9999)", type=int, default=9999)
	parser.add_argument("-ch", "--chunks", help="Specify the chunk size; 0 and below loads all files at once (default: 100)", type=int, default=100)
	parser.add_argument("-t", "--threads", help="Specify the number of threads to be used (default: 1)" , type=int, default=1)
	args = parser.parse_args()
	
	data_path = args.data
	dis_files = os.listdir(data_path)
	output = args.output
	os.makedirs(output, exist_ok=True)
	encoding = args.encoding
	code_path = args.codes
	code_map = args.mapping
	resamples = args.resamples if args.resamples > 1 else 1
	chunks = args.chunks if args.chunks > 0 else len(dis_files)
	threads = args.threads if args.threads > 0 else 1
	
	encoding_df = pd.read_csv(encoding, sep="\t", header=0, index_col=0).fillna("1")
	code_map_df = pd.read_csv(code_map, sep="\t", header=0, index_col=0)
	comb_dis_df = pd.DataFrame()
	
	frames = []
	with mp.Pool(processes=threads) as pool:
		for chunk in range(0, len(dis_files), chunks):
			dis_data = []
			chunked_files = dis_files[chunk:chunk+chunks]
			max_chunk = min(chunk+chunks, len(dis_files))
			for file in tqdm.tqdm(chunked_files, desc=f"Loading distribution files for chunk [{chunk}-{max_chunk}/{len(dis_files)}]"):
				tax_id = int(file.split(".csv")[0].split("_")[1])
				dis_df = pd.read_csv(os.path.join(data_path, file), sep="\t", header=0, index_col=0, on_bad_lines="skip")
				code_id = int(encoding_df.loc[tax_id, "GeneticID"])
				code_name = code_map_df.loc[code_id, "Name"]
				freq_funcs = None
				with open(os.path.join(code_path, f"{code_name}.yaml"), "r") as code_reader:
					gen_code = yaml.safe_load(code_reader)
					freq_funcs = ef.build_functions(gen_code)
				
				dis_data.append([tax_id, dis_df, code_name, freq_funcs, resamples])	
			
			result = list(tqdm.tqdm(pool.imap(combine_distribution_stats, dis_data), total=len(dis_data), desc=f"Calculating amino acid statistics for chunk " 
																											   f"[{chunk}-{max_chunk}/{len(dis_files)}]"))
			for res in result:
				frames.append(res)
					
	comb_dis_df = pd.concat(frames)
	comb_dis_df.astype(str).fillna("0.0", inplace=True)
	comb_dis_df.index.name = "TaxID"
	comb_dis_df.to_csv(os.path.join(output, "combined_distributions.csv"), sep="\t")
	
	### Summarize data
	corr_start_idx = [i for i,col in enumerate(comb_dis_df.columns) if col.startswith("Ps_")][0]
	# Summarize the mean and std data weighting them by the number of proteins
	summary_df = pd.DataFrame(index=["Mean", "Std", "Min", "Max"])
	cols = [col for col in comb_dis_df.columns if col.endswith("_mean")]
	for col in cols:
		std_col = col[:-4] + "std"
		weighted_mean = (comb_dis_df["#Proteins"] * comb_dis_df[col]).sum() / comb_dis_df["#Proteins"].sum()
		summary_df.loc["Mean", col] = weighted_mean
		sum_of_squares = (((comb_dis_df["#Proteins"] - 1) * comb_dis_df[std_col]**2) + (comb_dis_df["#Proteins"] * (comb_dis_df[col] - weighted_mean)**2)).sum()
		weighted_var = sum_of_squares / (comb_dis_df["#Proteins"].sum() - 1)
		summary_df.loc["Std", col] = np.sqrt(weighted_var)
		
	summary_df.loc["Min", cols] = comb_dis_df[cols].min()
	summary_df.loc["Max", cols] = comb_dis_df[cols].max()
	
	# Summarize the non-mean, non-std, non-correlation data
	cols = [col for col in comb_dis_df.columns[:corr_start_idx] if not col.endswith("_mean") and not col.endswith("_std")]
	summary_df.loc["Mean", cols] = comb_dis_df[cols].mean()
	summary_df.loc["Std", cols] = comb_dis_df[cols].std()
	summary_df.loc["Min", cols] = comb_dis_df[cols].min()
	summary_df.loc["Max", cols] = comb_dis_df[cols].max()
	
	# Summarize the correlation data
	cols = comb_dis_df.columns[corr_start_idx:-1]
	corr_means = []
	corr_stds = []
	for col in cols:
		mean,std = fisher_Z(comb_dis_df[col])
		corr_means.append(mean)
		corr_stds.append(std)
		
	summary_df.loc["Mean", cols] = corr_means
	summary_df.loc["Std", cols] = corr_stds
	summary_df.loc["Min", cols] = comb_dis_df[cols].min()
	summary_df.loc["Max", cols] = comb_dis_df[cols].max()
	summary_df.to_csv(os.path.join(output, "distribution_description.csv"), sep="\t")
	
		
