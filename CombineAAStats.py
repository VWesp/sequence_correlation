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


# Canonical amino acids order
amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I", "A", "G", "P", "T", "V", "L", "R", "S"]
code_cols = [f"{aa}_code" for aa in amino_acids]
gc_cols = [f"{aa}_gc" for aa in amino_acids]
	
# Full name for each amino acid
one_letter_code = {"M": "Methionine", "T": "Threonine", "N": "Asparagine", "K": "Lysine", "S": "Serine", "R": "Arginine", "V": "Valine", "A": "Alanine", "D": "Aspartic_acid",
				   "E": "Glutamic_acid", "G": "Glycine", "F": "Phenylalanine", "L": "Leucine", "Y": "Tyrosine", "C": "Cysteine", "W": "Tryptophane", "P": "Proline",
				   "H": "Histidine", "Q": "Glutamine", "I": "Isoleucine"}


def combine_distribution_stats(data):
	tax_id,dis_df,code_name,freq_funcs,resamples,repl = data
	
	dis_df.fillna(0.0, inplace=True)
	dis_sr = pd.Series(name=tax_id)
	dis_sr["#Proteins"] = len(dis_df)
	dis_sr["Length"] = dis_df["Length"].mean()
	dis_sr["GC"] = dis_df["GC"].mean()
	 # Load frequency functions for each amino acid based on the codons and independent of GC content (GC=50%)
	code_freq_func = ef.calculate_frequencies(freq_funcs, 0.5)
	 # Load frequency functions for each amino acid based on the codons and GC content
	gc_freq_func = ef.calculate_frequencies(freq_funcs, dis_sr["GC"])
	for aa in amino_acids:
		try:
			dis_sr[aa] = dis_df[aa].mean()
		except KeyError:
			dis_sr[aa] = repl
		
		###### Code frequency
		dis_sr[f"{aa}_code"] = code_freq_func["amino"][one_letter_code[aa]]
		###### GC frequency
		dis_sr[f"{aa}_gc"] = gc_freq_func["amino"][one_letter_code[aa]]
	
	######
	obs_val = np.asarray(dis_sr[amino_acids], dtype=float)
	obs_cls = skb.stats.composition.closure(obs_val)
	obs_clr = skb.stats.composition.clr(obs_cls)
	dis_sr[amino_acids] = obs_cls
	for index,aa in enumerate(amino_acids):
		dis_sr[f"{aa}_clr"] = obs_clr[index]
	
	######
	code_val = np.asarray(dis_sr[code_cols], dtype=float)
	code_cls = skb.stats.composition.closure(code_val)
	code_clr = skb.stats.composition.clr(code_cls)
	dis_sr[code_cols] = code_cls
	for index,aa in enumerate(amino_acids):
		dis_sr[f"{aa}_code_clr"] = code_clr[index]
		dis_sr[f"{aa}_code_clr_lr"] = np.log(obs_clr[index] / code_clr[index])
		
	######
	gc_val = np.asarray(dis_sr[gc_cols], dtype=float)
	gc_cls = skb.stats.composition.closure(gc_val)
	gc_clr = skb.stats.composition.clr(gc_cls)
	dis_sr[gc_cols] = gc_cls
	for index,aa in enumerate(amino_acids):
		dis_sr[f"{aa}_gc_clr"] = gc_clr[index]
		dis_sr[f"{aa}_gc_clr_lr"] = np.log(obs_clr[index] / gc_clr[index])
	
	############################ Aitchison distance between species observed and predicted amino acid frequencies
	dis_sr["aitchison_code"] = sci.spatial.distance.euclidean(obs_clr, code_clr)
	dis_sr["aitchison_gc"] = sci.spatial.distance.euclidean(obs_clr, gc_clr)
	############################ Pearson code
	corr_stats = sci.stats.permutation_test((obs_clr,), lambda x: sci.stats.pearsonr(x, code_clr).statistic, permutation_type="pairings", n_resamples=resamples)
	dis_sr["Ps_code"] = corr_stats.statistic
	dis_sr["Ps_code_p"] = corr_stats.pvalue
	############################ Pearson frequency
	corr_stats = sci.stats.permutation_test((obs_clr,), lambda x: sci.stats.pearsonr(x, gc_clr).statistic, permutation_type="pairings", n_resamples=resamples)
	dis_sr["Ps_gc"] = corr_stats.statistic
	dis_sr["Ps_gc_p"] = corr_stats.pvalue
	############################ Spearman code
	corr_stats = sci.stats.permutation_test((obs_clr,), lambda x: sci.stats.spearmanr(x, code_clr).statistic, permutation_type="pairings", n_resamples=resamples)
	dis_sr["Sm_code"] = corr_stats.statistic
	dis_sr["Sm_code_p"] = corr_stats.pvalue
	############################ Spearman frequency
	corr_stats = sci.stats.permutation_test((obs_clr,), lambda x: sci.stats.spearmanr(x, gc_clr).statistic, permutation_type="pairings", n_resamples=resamples)
	dis_sr["Sm_gc"] = corr_stats.statistic
	dis_sr["Sm_gc_p"] = corr_stats.pvalue
	############################ Kendall tau code
	corr_stats = sci.stats.permutation_test((obs_clr,), lambda x: sci.stats.kendalltau(x, code_clr).statistic, permutation_type="pairings", n_resamples=resamples)
	dis_sr["Kt_code"] = corr_stats.statistic
	dis_sr["Kt_code_p"] = corr_stats.pvalue
	############################ Kendall tau frequency
	corr_stats = sci.stats.permutation_test((obs_clr,), lambda x: sci.stats.kendalltau(x, gc_clr).statistic, permutation_type="pairings", n_resamples=resamples)
	dis_sr["Kt_gc"] = corr_stats.statistic
	dis_sr["Kt_gc_p"]  = corr_stats.pvalue
		
	dis_sr["Genetic_code"] = code_name													
	return dis_sr.to_frame().T


# main method
if __name__ == "__main__":
	mp.freeze_support()
	
	parser = argparse.ArgumentParser(description="Compute amino acid distributions in proteomes")
	parser.add_argument("-d", "--data", help="Specify the path to the folder with the distribution files", required=True)
	parser.add_argument("-o", "--output", help="Specify the output file", required=True)
	parser.add_argument("-e", "--encoding", help="Set the path to the encoding file", required=True)
	parser.add_argument("-c", "--codes", help="Specify the path to the folder with the genetic code files", required=True)
	parser.add_argument("-m", "--mapping", help="Set the path to the mappings of the genetic codes", required=True)
	parser.add_argument("-r", "--resamples", help="Specify the number of resamples for the permutation tests (default: 9999)", type=int, default=9999)
	parser.add_argument("-rp", "--replace", help="Replace zero values with a small number (default: 1e-12)", type=float, default=1e-12)
	parser.add_argument("-ch", "--chunks", help="Specify the chunk size; 0 and below loads all files at once (default: 100)", type=int, default=100)
	parser.add_argument("-t", "--threads", help="Specify the number of threads to be used (default: 1)" , type=int, default=1)
	args = parser.parse_args()
	
	data_path = args.data
	dis_files = os.listdir(data_path)
	output = args.output
	encoding = args.encoding
	code_path = args.codes
	code_map = args.mapping
	resamples = args.resamples if args.resamples > 1 else 1
	repl = args.replace
	chunks = args.chunks if args.chunks > 0 else len(dis_files)
	threads = args.threads if args.threads > 0 else 1
	
	encoding_df = pd.read_csv(encoding, sep="\t", header=0, index_col=0).fillna("1")
	code_map_df = pd.read_csv(code_map, sep="\t", header=0, index_col=0)
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
				
				dis_data.append([tax_id, dis_df, code_name, freq_funcs, resamples, repl])	
			
			result = list(tqdm.tqdm(pool.imap(combine_distribution_stats, dis_data), total=len(dis_data), desc=f"Calculating amino acid statistics for chunk " 
																											   f"[{chunk}-{max_chunk}/{len(dis_files)}]"))
			for res in result:
				frames.append(res)
					
	comb_dis_df = pd.concat(frames)
	comb_dis_df.astype(str).fillna("0.0", inplace=True)
	comb_dis_df.index.name = "TaxID"
	comb_dis_df.to_csv(output, sep="\t")
	
		
