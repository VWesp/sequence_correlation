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
	func_df = pd.Series(name=tax_id, index=["Genetic_code"]+amino_acids)
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
	dist_df[amino_acids] = obs_clr - code_clr
	dist_df["Aitchison_distance"] = sci.spatial.distance.euclidean(data_x, data_y)
	return dist_df.to_frame().T


def p_value(obs, perms):
	return (np.sum(np.abs(perms) >= np.abs(obs)) + 1) / (len(perms) + 1)


def calculate_corr_func(func, obs, obs_perms, pred):
	corr = func(obs, pred).statistic
	corr_perms = np.array([func(perm, pred).statistic for perm in obs_perms])
	p_value = p_value(corr, corr_perms)
	return corr, p_value


def correlate_data(tax_id, obs, pred, resamples):
	# Permutate observed values
	perms = np.array([rng.permutation(len(obs)) for _ in range(resamples)])
	obs_perms = obs[perms]

	code_corr_df = pd.Series(name=tax_id, index=cols)
	# Pearson
	code_corr_df["Pearson"] = sci.stats.pearsonr(obs_clr, code_clr).statistic
	pears_perms = np.array([sci.stats.pearsonr(perm, code_clr).statistic for perm in obs_perms])
	code_corr_df["Pearson_p"] = p_value(code_corr_df["Pearson"], pears_perms)


	corr_df = pd.Series(name=tax_id, index=["Pearson", "Pearson_p", "Pearson_q",
											"Spearman", "Spearman_p", "Spearman_q",
											"Kendall", "Kendall_p", "Kendall_q"])
	# Pearson
	corr_df["Pearson", "Pearson_p"] = calculate_corr_func(sci.stats.pearsonr, obs, obs_perms, pred)
	# Spearman
	corr_df["Spearman", "Spearman_p"] = calculate_corr_func(sci.stats.spearmanr, obs, obs_perms, pred)
	# Kendall
	corr_df["Kendall", "Kendall_p"] = calculate_corr_func(sci.stats.kendalltau, obs, obs_perms, pred)
	return corr_df.to_frame().T



def compare_values(tax_id, freq_funcs, gc, code_name, obs_clr, obs_ilr, obs_alr, resamples):
	# Load frequency functions for each amino acid based on the codons and GC content
	pred_df, pred_cls = load_freq_funcs(tax_id, freq_funcs, gc, code_name)

	# CLR values
	clr_df, pred_clr = transform_data(tax_id, skb.stats.composition.clr, pred_cls)
	# CLR distances and Aitchison distance
	clr_delta_df = calculate_distance(tax_id, obs_clr, pred_clr)
	# CLR correlations
	clr_corr_df = correlate_data(tax_id, obs_clr, pred_clr, resamples)

	# ILR values
	ilr_df, pred_ilr = transform_data(tax_id, skb.stats.composition.ilr, pred_cls)
	# ILR distances and Aitchison distance
	ilr_delta_df = calculate_distance(tax_id, obs_ilr, pred_ilr)
	# ILR correlations
	ilr_corr_df = correlate_data(tax_id, obs_ilr, pred_ilr, resamples)

	# ALR values
	alr_df, pred_alr = transform_data(tax_id, skb.stats.composition.alr, pred_cls)
	# ALR distances and Aitchison distance
	alr_delta_df = calculate_distance(tax_id, obs_alr, pred_alr)
	# ALR correlations
	alr_corr_df = correlate_data(tax_id, obs_alr, pred_alr, resamples)

	return [pred_df, clr_df, clr_delta_df, clr_corr_df,
			ilr_df, ilr_delta_df, ilr_corr_df,
			alr_df, alr_delta_df, alr_corr_df]



def combine_distribution_stats(data):
	tax_id,dis_df,code_name,freq_funcs,resamples,replace,rng = data
	
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
	obs_clr_df, obs_clr = transform_data(skb.stats.composition.clr, tax_id, obs_cls)
	# Observed ILR values
	obs_ilr_df, obs_ilr = transform_data(skb.stats.composition.ilr, tax_id, obs_cls)
	# Observed ALR values
	obs_alr_df, obs_alr = transform_data(skb.stats.composition.alr, tax_id, obs_cls)
	return_dfs = [obs_df.to_frame().T, obs_clr_df, obs_ilr_df, obs_alr_df]

	# Compare observed and code frequencies
	code_dfs = compare_values(tax_id, freq_funcs, 0.5, code_name, obs_clr, obs_ilr, obs_alr, resamples)
	return_dfs.extend(code_dfs)

	# Compare observed and code+GC frequencies
	gc_dfs = compare_values(tax_id, freq_funcs, float(obs_df["GC"]), code_name, obs_clr, obs_ilr, obs_alr, resamples)
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
				with open(os.path.join(code_path, f"{code_name}.yaml"), "r") as code_reader:
					gen_code = yaml.safe_load(code_reader)
					freq_funcs = ef.build_functions(gen_code)
				
				dis_data.append([tax_id, dis_df, code_name, freq_funcs, resamples, replace, rng])	
			
			result = list(tqdm.tqdm(pool.imap(combine_distribution_stats, dis_data), total=len(dis_data), desc=f"Calculating amino acid statistics for chunk " 
																											   f"[{chunk}-{max_chunk}/{len(dis_files)}]"))
			for res in result:
				frames.append(res)

	file_list = ["obs_freq", "obs_clr", "obs_ilr", "obs_alr",
				 "code_freq", "code_clr", "code_clr_delta", "code_clr_corr",
				 "code_ilr", "code_ilr_delta", "code_ilr_corr",
				 "code_alr", "code_alr_delta", "code_alr_corr",
				 "gc_freq", "gc_clr", "gc_clr_delta", "gc_clr_corr",
				 "gc_ilr", "gc_ilr_delta", "gc_ilr_corr",
				 "gc_alr", "gc_alr_delta", "gc_alr_corr"]
	for index,file in enumerate(file_list):
		comb_df = pd.concat([f[index] for f in frames])
		comb_df.index.name = "TaxID"
		if(file.endswith("_corr")):
			comb_df["Pearson_q"] = sm.multipletests(comb_df["Pearson_p"], method="fdr_bh")[1]
			comb_df["Spearman_q"] = sm.multipletests(comb_df["Spearman_p"], method="fdr_bh")[1]
			comb_df["Kendall_q"] = sm.multipletests(comb_df["Kendall_p"], method="fdr_bh")[1]

		comb_df.to_csv(os.path.join(output, f"{file}.csv"), sep="\t")
	
		
