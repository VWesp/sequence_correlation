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
	obs_clr_df = pd.Series(name=tax_id, index=amino_acids)
	obs_clr = np.array(skb.stats.composition.clr(obs_cls))
	obs_clr_df[amino_acids] = obs_clr

	# Load frequency functions for each amino acid based on the codons and independent of GC content (GC=50%)
	code_df = pd.Series(name=tax_id, index=["Genetic_code"]+amino_acids, dtype=str)
	code_df["Genetic_code"] = code_name	
	code_freq_func = ef.calculate_frequencies(freq_funcs, 0.5)
	code_freq_aas = np.array([code_freq_func["amino"][one_letter_code[aa]] for aa in amino_acids])
	code_cls = np.array(skb.stats.composition.closure(code_freq_aas))
	code_df[amino_acids] = code_cls
	# Code CLR values
	code_clr_df = pd.Series(name=tax_id, index=amino_acids)
	code_clr = np.array(skb.stats.composition.clr(code_cls))
	code_clr_df[amino_acids] = code_clr
	# CLR distances and Aitchison distance
	code_delta_df = pd.Series(name=tax_id, index=amino_acids+["Aitchison_distance"])
	code_delta_df[amino_acids] = obs_clr - code_clr
	code_delta_df["Aitchison_distance"] = sci.spatial.distance.euclidean(obs_clr, code_clr)

	# Load frequency functions for each amino acid based on the codons and GC content
	gc_df = pd.Series(name=tax_id, index=["Genetic_code"]+amino_acids, dtype=str)
	gc_df["Genetic_code"] = code_name
	gc_freq_func = ef.calculate_frequencies(freq_funcs, obs_df["GC"])
	gc_freq_aas = np.array([gc_freq_func["amino"][one_letter_code[aa]] for aa in amino_acids])
	gc_cls = np.array(skb.stats.composition.closure(gc_freq_aas))
	gc_df[amino_acids] = gc_cls
	# Code+GC content CLR values and Aitchison distance
	gc_clr_df = pd.Series(name=tax_id, index=amino_acids)
	gc_clr = np.array(skb.stats.composition.clr(gc_cls))
	gc_clr_df[amino_acids] = gc_clr
	# CLR distances and Aitchison distance
	gc_delta_df = pd.Series(name=tax_id, index=amino_acids+["Aitchison_distance"])
	gc_delta_df[amino_acids] = obs_clr - gc_clr
	gc_delta_df["Aitchison_distance"] = sci.spatial.distance.euclidean(obs_clr, gc_clr)
	
	# Correlation values
	perms = np.array([rng.permutation(len(obs_clr)) for _ in range(resamples)])
	obs_perms = obs_clr[perms]

	def p_value(obs, perms):
		return (np.sum(np.abs(perms) >= np.abs(obs)) + 1) / (len(perms) + 1)

	cols = ["Pearson", "Pearson_p", "Pearson_q", "Spearman", "Spearman_p", "Spearman_q", "Kendall", "Kendall_p", "Kendall_q"]
	# for code frequencies
	code_corr_df = pd.Series(name=tax_id, index=cols)
	# Pearson
	code_corr_df["Pearson"] = sci.stats.pearsonr(obs_clr, code_clr).statistic
	pears_perms = np.array([sci.stats.pearsonr(perm, code_clr).statistic for perm in obs_perms])
	code_corr_df["Pearson_p"] = p_value(code_corr_df["Pearson"], pears_perms)
	# Spearman
	code_corr_df["Spearman"] = sci.stats.spearmanr(obs_clr, code_clr).statistic
	spearm_perms = np.array([sci.stats.spearmanr(perm, code_clr).statistic for perm in obs_perms])
	code_corr_df["Spearman_p"] = p_value(code_corr_df["Spearman"], spearm_perms)
	# Kendall
	code_corr_df["Kendall"] = sci.stats.kendalltau(obs_clr, code_clr).statistic
	kendallt_perms = np.array([sci.stats.kendalltau(perm, code_clr).statistic for perm in obs_perms])
	code_corr_df["Kendall_p"] = p_value(code_corr_df["Kendall"], kendallt_perms)

	# for code+GC content frequencies
	gc_corr_df = pd.Series(name=tax_id, index=cols)
	# Pearson
	gc_corr_df["Pearson"] = sci.stats.pearsonr(obs_clr, gc_clr).statistic
	pears_perms = np.array([sci.stats.pearsonr(perm, gc_clr).statistic for perm in obs_perms])
	gc_corr_df["Pearson_p"] = p_value(gc_corr_df["Pearson"], pears_perms)
	# Spearman
	gc_corr_df["Spearman"] = sci.stats.spearmanr(obs_clr, gc_clr).statistic
	spearm_perms = np.array([sci.stats.spearmanr(perm, gc_clr).statistic for perm in obs_perms])
	gc_corr_df["Spearman_p"] = p_value(gc_corr_df["Spearman"], spearm_perms)
	# Kendall
	gc_corr_df["Kendall"] = sci.stats.kendalltau(obs_clr, gc_clr).statistic
	kendallt_perms = np.array([sci.stats.kendalltau(perm, gc_clr).statistic for perm in obs_perms])
	gc_corr_df["Kendall_p"] = p_value(gc_corr_df["Kendall"], kendallt_perms)

											
	return [obs_df.to_frame().T, obs_clr_df.to_frame().T,
			code_df.to_frame().T, code_clr_df.to_frame().T, code_delta_df.to_frame().T,
			gc_df.to_frame().T, gc_clr_df.to_frame().T, gc_delta_df.to_frame().T,
			code_corr_df.to_frame().T, gc_corr_df.to_frame().T]


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

	file_list = ["obs_frequencies", "obs_clr",
				 "code_frequencies", "code_clr", "code_clr_delta",
				 "gc_frequencies", "gc_clr", "gc_clr_delta",
				 "code_corrs", "gc_corrs"]
	for index,file in enumerate(file_list):
		comb_df = pd.concat([f[index] for f in frames])
		comb_df.index.name = "TaxID"
		if(file == "code_corrs" or file == "gc_corrs"):
			comb_df["Pearson_q"] = sm.multipletests(comb_df["Pearson_p"], method="fdr_bh")[1]
			comb_df["Spearman_q"] = sm.multipletests(comb_df["Spearman_p"], method="fdr_bh")[1]
			comb_df["Kendall_q"] = sm.multipletests(comb_df["Kendall_p"], method="fdr_bh")[1]

		comb_df.to_csv(os.path.join(output, f"{file}.csv"), sep="\t")
	
		
