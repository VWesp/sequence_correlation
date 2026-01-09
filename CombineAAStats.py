import os
import tqdm
import yaml
import argparse
import numpy as np
import pandas as pd
import scipy as sci
import skbio as skb
import multiprocessing as mp
import equation_functions as ef
import statsmodels.stats.multitest as sm


# Canonical amino acids order
amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I", "A", "G", "P", "T", "V", "L", "R", "S"]
# Full name for each amino acid
one_letter_code = {"M": "Methionine", "T": "Threonine", "N": "Asparagine", "K": "Lysine", "S": "Serine", "R": "Arginine", "V": "Valine", "A": "Alanine", "D": "Aspartic_acid",
				   "E": "Glutamic_acid", "G": "Glycine", "F": "Phenylalanine", "L": "Leucine", "Y": "Tyrosine", "C": "Cysteine", "W": "Tryptophane", "P": "Proline",
				   "H": "Histidine", "Q": "Glutamine", "I": "Isoleucine"}


def load_freq_funcs(tax_id, func, gc, code_name):
	func_df = pd.Series(name=tax_id, index=["Genetic_code"]+amino_acids, dtype=str)
	func_df["Genetic_code"] = code_name	
	freq_func = ef.calculate_frequencies(func, gc)
	freq_func_aas = np.array([freq_func["amino"][one_letter_code[aa]] for aa in amino_acids])
	freq_cls = np.array(skb.stats.composition.closure(freq_func_aas))
	func_df[amino_acids] = freq_cls
	return func_df.to_frame().T, freq_cls


def transform_data(tax_id, func, data):
	trans_df = pd.Series(name=tax_id, index=amino_acids)
	trans_data = np.array(func(data))
	trans_df[amino_acids] = trans_data
	return trans_df.to_frame().T, trans_data


def calculate_distance(tax_id, data_x, data_y):
	dist_df = pd.Series(name=tax_id, index=amino_acids+["Aitchison_distance"])
	dist_df[amino_acids] = data_x - data_y
	dist_df["Aitchison_distance"] = np.sqrt((dist_df[amino_acids]**2).sum())
	return dist_df.to_frame().T


def calculate_p_value(obs, perms):
	return (np.sum(np.abs(perms) >= np.abs(obs)) + 1) / (len(perms) + 1)


def calculate_corr_func(func, obs, obs_perms, pred):
	corr = func(obs, pred).statistic
	corr_perms = np.array([func(perm, pred).statistic for perm in obs_perms])
	p_value = calculate_p_value(corr, corr_perms)
	return corr, p_value


def correlate_data(tax_id, obs, pred, resamples):
	# Permutate observed values
	perms = np.array([rng.permutation(len(obs)) for _ in range(resamples)])
	obs_perms = obs[perms]

	corr_df = pd.Series(name=tax_id, index=["Pearson", "Pearson_p", "Pearson_q",
											"Spearman", "Spearman_p", "Spearman_q",
											"Kendall", "Kendall_p", "Kendall_q"])
	# Pearson
	corr_df[["Pearson", "Pearson_p"]] = calculate_corr_func(sci.stats.pearsonr, obs, obs_perms, pred)
	# Spearman
	corr_df[["Spearman", "Spearman_p"]] = calculate_corr_func(sci.stats.spearmanr, obs, obs_perms, pred)
	# Kendall
	corr_df[["Kendall", "Kendall_p"]] = calculate_corr_func(sci.stats.kendalltau, obs, obs_perms, pred)
	return corr_df.to_frame().T


# Taken from: https://file.statistik.tuwien.ac.at/filz/papers/CorrMatG09.pdf
def calculate_balances(tax_id, data, codon_parts):
	balances = {}

	def calculate_splits(data, codon_parts):
		codons = sorted(set(codon_parts))
		if(len(codons) == 1):
			return

		mid = len(codons) // 2

		r_codons = codons[mid:]
		split = [i for i,e in enumerate(codon_parts) if e in r_codons]
		r_data = data[split]
		r = len(r_data)
		r_mean = np.mean(np.log(r_data))

		s_codons = codons[:mid]
		mask = np.ones(len(codon_parts), dtype=bool)
		mask[split] = False
		s_data = data[mask]
		s = len(s_data)
		s_mean = np.mean(np.log(s_data))

		balance_name = f"{",".join([str(i) for i in r_codons])}_vs_{",".join([str(i) for i in s_codons])}"
		scale = np.sqrt((r * s) / (r + s))
		balances[balance_name] = scale * (r_mean - s_mean)
		calculate_splits(r_data, codon_parts[split])
		calculate_splits(s_data, codon_parts[mask])

	calculate_splits(data, codon_parts)
	bal_df = pd.Series(data=list(balances.values()), name=tax_id, index=list(balances.keys()))
	return bal_df.to_frame().T


def compare_values(tax_id, freq_funcs, gc, code_name, obs_clr, obs_bal_df, codon_parts, resamples):
	# Load frequency functions for each amino acid based on the codons and GC content
	pred_df, pred_cls = load_freq_funcs(tax_id, freq_funcs, gc, code_name)

	# Calculate the balances between high codon amino acids and low codons
	# as well as their distance to the observed balances
	pred_bal_df = calculate_balances(tax_id, pred_cls, codon_parts)
	pred_bal_delta_df = obs_bal_df - pred_bal_df
	pred_bal_delta_df["Aitchison_distance"] = np.sqrt((pred_bal_delta_df**2).sum(axis=1))

	# CLR values
	clr_df, pred_clr = transform_data(tax_id, skb.stats.composition.clr, pred_cls)
	# CLR distances and Aitchison distance
	clr_delta_df = calculate_distance(tax_id, obs_clr, pred_clr)
	# CLR correlations
	clr_corr_df = correlate_data(tax_id, obs_clr, pred_clr, resamples)

	return [pred_df, pred_bal_df, pred_bal_delta_df, clr_df, clr_delta_df, clr_corr_df]


def combine_distribution_stats(data):
	tax_id,dis_df,code_name,freq_funcs,codon_parts,resamples,replace,rng = data
	
	dis_df.fillna(0.0, inplace=True)

	# Observed frequencies
	obs_df = pd.Series(name=tax_id, index=["#Proteins", "Length", "GC"]+amino_acids)
	obs_df["#Proteins"] = len(dis_df)
	obs_df["Length"] = dis_df["Length"].median()
	obs_df["GC"] = dis_df["GC"].median()
	obs_median_aas = np.array([0.0] * len(amino_acids))
	for index,aa in enumerate(amino_acids):
		if aa in dis_df.columns:
			obs_median_aas[index] = dis_df[aa].median()

	obs_median_aas[obs_median_aas == 0] = replace
	obs_cls = np.array(skb.stats.composition.closure(obs_median_aas))
	obs_df[amino_acids] = obs_cls
	# Observed CLR values
	obs_clr_df, obs_clr = transform_data(tax_id, skb.stats.composition.clr, obs_cls)
	# Calculate the balances between high codon amino acids and low codons
	obs_bal_df = calculate_balances(tax_id, obs_cls, codon_parts)
	return_dfs = [obs_df.to_frame().T, obs_bal_df, obs_clr_df]

	# Compare observed and code frequencies
	code_dfs = compare_values(tax_id, freq_funcs, 0.5, code_name, obs_clr, obs_bal_df, codon_parts, resamples)
	return_dfs.extend(code_dfs)

	# Compare observed and code+GC frequencies
	gc_dfs = compare_values(tax_id, freq_funcs, float(obs_df["GC"]), code_name, obs_clr, obs_bal_df, codon_parts, resamples)
	return_dfs.extend(gc_dfs)
											
	return return_dfs


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
	parser.add_argument("-rp", "--replace", help="Replace a (small) value to zero values for the CLR-transformation (default: 1e-12)", type=float, default=1e-12)
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
	replace = args.replace
	chunks = args.chunks if args.chunks > 0 else len(dis_files)
	threads = args.threads if args.threads > 0 else 1
	
	encoding_df = pd.read_csv(encoding, sep="\t", header=0, index_col=0).fillna("1")
	code_map_df = pd.read_csv(code_map, sep="\t", header=0, index_col=0)
	rng = np.random.default_rng()
	frames = []
	with mp.Pool(processes=threads) as pool:
		for chunk in range(0, len(dis_files), chunks):
			dis_data = []
			chunked_files = dis_files[chunk:chunk+chunks]
			max_chunk = min(chunk+chunks, len(dis_files))
			for file in tqdm.tqdm(chunked_files, desc=f"Loading distribution files for chunk [{chunk}-{max_chunk}/{len(dis_files)}]"):
				tax_id = int(file.split(".csv")[0].split("_")[1])
				dis_df = pd.read_csv(os.path.join(data_path, file), sep="\t", header=0, index_col=0, on_bad_lines="skip").drop(["Name", "Status"], axis=1)
				code_id = int(encoding_df.loc[tax_id, "GeneticID"])
				code_name = code_map_df.loc[code_id, "Name"]
				freq_funcs = None
				codon_parts = None
				with open(os.path.join(code_path, f"{code_name}.yaml"), "r") as code_reader:
					gen_code = yaml.safe_load(code_reader)
					freq_funcs = ef.build_functions(gen_code)
					codon_parts = np.array([len(gen_code[one_letter_code[aa]]) for aa in amino_acids])

				dis_data.append([tax_id, dis_df, code_name, freq_funcs, codon_parts, resamples, replace, rng])	
			
			result = list(tqdm.tqdm(pool.imap(combine_distribution_stats, dis_data), total=len(dis_data), desc=f"Calculating amino acid statistics for chunk " 
																											   f"[{chunk}-{max_chunk}/{len(dis_files)}]"))
			for res in result:
				frames.append(res)	

	file_list = ["obs_freq", "obs_bal", "obs_clr",
				 "code_freq", "code_bal", "code_bal_delta", "code_clr", "code_clr_delta", "code_clr_corr",
				 "gc_freq", "gc_bal", "gc_bal_delta", "gc_clr", "gc_clr_delta", "gc_clr_corr"]
	for index,file in enumerate(file_list):
		comb_df = pd.concat([f[index] for f in frames])
		comb_df.index.name = "TaxID"
		if(file.endswith("_corr")):
			comb_df["Pearson_q"] = sm.multipletests(comb_df["Pearson_p"], method="fdr_bh")[1]
			comb_df["Spearman_q"] = sm.multipletests(comb_df["Spearman_p"], method="fdr_bh")[1]
			comb_df["Kendall_q"] = sm.multipletests(comb_df["Kendall_p"], method="fdr_bh")[1]

		comb_df.to_csv(os.path.join(output, f"{file}.csv"), sep="\t")
	
		
