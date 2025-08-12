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


def fisher_Z(x):
	### Fisher's Z-transformation
	# Correlations
	z = [0.5*np.log((1+r)/(1-r)) for r in x]
	mean_z = np.mean(z)
	std_z = 1 / np.sqrt(len(x)-3)
	mean_r = (np.exp(2*mean_z)-1) / (np.exp(2*mean_z)+1)
	std_r = (np.exp(2*std_z)-1) / (np.exp(2*std_z)+1)
	return [mean_r, std_r]


def combine_distribution_stats(data):
	tax_id,dis_df,code_name,freq_funcs,resamples = data

	# Canonical amino acids order
	amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I", "A", "G", "P", "T", "V", "L", "R", "S"]
	
	# One letter code for the amino acids
	one_letter_code = {"M": "Methionine", "T": "Threonine", "N": "Asparagine", "K": "Lysine", "S": "Serine", "R": "Arginine", "V": "Valine", "A": "Alanine", "D": "Aspartic_acid", 
						"E": "Glutamic_acid", "G": "Glycine", "F": "Phenylalanine", "L": "Leucine", "Y": "Tyrosine", "C": "Cysteine", "W": "Tryptophane", "P": "Proline",
						"H": "Histidine", "Q": "Glutamine", "I": "Isoleucine"}

	dis_df.fillna(0.0, inplace=True)
	dis_sr = pd.Series(name=tax_id)
	dis_sr["#Proteins"] = len(dis_df)
	dis_sr["Length_median"] = dis_df["Length"].median()
	dis_sr["Length_mad"] = (dis_df["Length"] - dis_sr["Length_median"]).abs().median()
	dis_sr["GC_median"] = dis_df["GC"].median()
	dis_sr["GC_mad"] = (dis_df["GC"] - dis_sr["GC_median"]).abs().median()
	 # Load frequency functions for each amino acid based on the codons and independent of GC content (GC=50%)
	code_freq_func = ef.calculate_frequencies(freq_funcs, 0.5)
	 # Load frequency functions for each amino acid based on the codons and GC content
	gc_freq_func = ef.calculate_frequencies(freq_funcs, dis_sr["GC_median"])
	for aa in amino_acids:
		try:
			dis_sr[f"{aa}_median"] = dis_df[aa].median()
			dis_sr[f"{aa}_mad"] = (dis_df[aa] - dis_sr[f"{aa}_median"]).abs().median()
		except KeyError:
			dis_sr[f"{aa}_median"] = 0.0
			dis_sr[f"{aa}_mad"] = 0.0
		
		fold_nom = dis_sr[f"{aa}_median"] if dis_sr[f"{aa}_median"] > 0 else 1e-10
		############################ log-Fold change and percentage change between empirical and code data
		dis_sr[f"{aa}_code"] = code_freq_func["amino"][one_letter_code[aa]]
		dis_sr[f"{aa}_code_lfc"] = np.log(fold_nom / dis_sr[f"{aa}_code"])
		dis_sr[f"{aa}_code_pct"] = (dis_sr[f"{aa}_median"] - dis_sr[f"{aa}_code"]) / dis_sr[f"{aa}_code"]
		############################ log-Fold change and percentage change between empirical and frequency data
		dis_sr[f"{aa}_gc"] = gc_freq_func["amino"][one_letter_code[aa]]
		dis_sr[f"{aa}_gc_lfc"] = np.log(fold_nom / dis_sr[f"{aa}_gc"])
		dis_sr[f"{aa}_gc_pct"] = (dis_sr[f"{aa}_median"] - dis_sr[f"{aa}_gc"]) / dis_sr[f"{aa}_gc"]
	

	aa_median_cols = [f"{aa}_median" for aa in amino_acids]
	code_freq_cols = [f"{aa}_code" for aa in amino_acids]
	gc_freq_cols = [f"{aa}_gc" for aa in amino_acids]
	############################ log-RMSE
	dis_sr["log_RMSE_code"] = np.log(skl.root_mean_squared_error(dis_sr[aa_median_cols], dis_sr[code_freq_cols]))
	dis_sr["log_RMSE_gc"] = np.log(skl.root_mean_squared_error(dis_sr[aa_median_cols], dis_sr[gc_freq_cols]))
	############################ Pearson code
	dis_sr["Ps_code"] = sci.pearsonr(dis_sr[aa_median_cols], dis_sr[code_freq_cols]).statistic
	dis_sr["Ps_code_p"] = sci.permutation_test((dis_sr[aa_median_cols],), lambda x: sci.pearsonr(x, dis_sr[code_freq_cols]).statistic, permutation_type="pairings", 
																					 			 n_resamples=resamples).pvalue
	############################ Pearson frequency
	dis_sr["Ps_gc"] = sci.pearsonr(dis_sr[aa_median_cols], dis_sr[gc_freq_cols]).statistic
	dis_sr["Ps_gc_p"] = sci.permutation_test((dis_sr[aa_median_cols],), lambda x: sci.pearsonr(x, dis_sr[gc_freq_cols]).statistic, permutation_type="pairings", 
																							   n_resamples=resamples).pvalue
	############################ Spearman code
	dis_sr["Sm_code"] = sci.spearmanr(dis_sr[aa_median_cols], dis_sr[code_freq_cols]).statistic
	dis_sr["Sm_code_p"] = sci.permutation_test((dis_sr[aa_median_cols],), lambda x: sci.spearmanr(x, dis_sr[code_freq_cols]).statistic, permutation_type="pairings", 
																					 			  n_resamples=resamples).pvalue
	############################ Spearman frequency
	dis_sr["Sm_gc"] = sci.spearmanr(dis_sr[aa_median_cols], dis_sr[gc_freq_cols]).statistic
	dis_sr["Sm_gc_p"] = sci.permutation_test((dis_sr[aa_median_cols],), lambda x: sci.spearmanr(x, dis_sr[gc_freq_cols]).statistic, permutation_type="pairings", 
																								n_resamples=resamples).pvalue
	############################ Kendall tau code
	dis_sr["Kt_code"] = sci.kendalltau(dis_sr[aa_median_cols], dis_sr[code_freq_cols], nan_policy="raise").statistic
	dis_sr["Kt_code_p"] = sci.permutation_test((dis_sr[aa_median_cols],), lambda x: sci.kendalltau(x, dis_sr[code_freq_cols]).statistic, permutation_type="pairings", 
																					  			 n_resamples=resamples).pvalue
	############################ Kendall tau frequency
	dis_sr["Kt_gc"] = sci.kendalltau(dis_sr[aa_median_cols], dis_sr[gc_freq_cols], nan_policy="raise").statistic
	dis_sr["Kt_gc_p"]  = sci.permutation_test((dis_sr[aa_median_cols],), lambda x: sci.kendalltau(x, dis_sr[gc_freq_cols]).statistic, permutation_type="pairings",
																								  n_resamples=resamples).pvalue
	
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
	parser.add_argument("-ch", "--chunks", help="Specify the chunk size; 0 loads all files at once (default: 100)", type=int, default=100)
	parser.add_argument("-t", "--threads", help="Specify the number of threads to be used (default: 1)" , type=int, default=1)
	args = parser.parse_args()
	
	data_path = args.data
	output = args.output
	encoding = args.encoding
	code_path = args.codes
	code_map = args.mapping
	resamples = args.resamples
	chunk_size = args.chunks
	threads = args.threads
	
	os.makedirs(output, exist_ok=True)
	
	encoding_df = pd.read_csv(encoding, sep="\t", header=0, index_col=0).fillna("1")
	code_map_df = pd.read_csv(code_map, sep="\t", header=0, index_col=0)
	dis_files = os.listdir(data_path)
	comb_dis_df = pd.DataFrame()
	if(chunk_size <= 0):
		chunk_size = len(dis_files)
	
	frames = []
	with mp.Pool(processes=threads) as pool:
		for chunk in range(0, len(dis_files), chunk_size):
			dis_data = []
			chunked_files = dis_files[chunk:chunk+chunk_size]
			max_chunk = min(chunk+chunk_size, len(dis_files))
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
	# Summarize non correlation data
	summary_df = pd.DataFrame(index=["Sum", "Median", "MAD", "Min", "Max", "25%", "75%"])
	cols = [col for col in comb_dis_df.columns[:corr_start_idx] if(not col.endswith("_mad"))]
	summary_df.loc["Sum", cols] = comb_dis_df[cols].sum()
	summary_df.loc["Median", cols] = comb_dis_df[cols].median()
	summary_df.loc["MAD", cols] = (comb_dis_df[cols] - summary_df.loc["Median", cols]).abs().median()
	summary_df.loc["Min", cols] = comb_dis_df[cols].min()
	summary_df.loc["Max", cols] = comb_dis_df[cols].max()
	summary_df.loc["25%", cols] = comb_dis_df[cols].quantile(0.25)
	summary_df.loc["75%", cols] = comb_dis_df[cols].quantile(0.75)
	summary_df.to_csv(os.path.join(output, "distribution_description.csv"), sep="\t")
		
	# Summarize correlation data
	corr_summary_df = pd.DataFrame(index=["Sum", "Median"])
	cols = comb_dis_df.columns[corr_start_idx:-1]
	for col in cols:
		mean,std = fisher_Z(comb_dis_df[col])
		corr_summary_df.loc["Mean", col] = mean
		corr_summary_df.loc["Std", col] = std
		
	corr_summary_df.to_csv(os.path.join(output, "correlation_description.csv"), sep="\t")
	
		
