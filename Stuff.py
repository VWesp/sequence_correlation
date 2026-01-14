import os
import yaml
import numpy as np
import pandas as pd
import equation_functions as ef

if __name__=="__main__":
	map_df = pd.read_csv("code_map.csv", sep="\t", header=0, index_col=2)

	codes = "genetic_codes"
	stand_freqs = None
	with open(os.path.join(codes, "standard.yaml"), "r") as reader:
		standard = yaml.safe_load(reader)
		freq_funcs = ef.build_functions(standard)
		freqs = ef.calculate_frequencies(freq_funcs, 0.5)["amino"]
		stand_freqs = np.array(list(dict(sorted(freqs.items())).values()))

	results = {}
	for file in os.listdir(codes):
		if(file != "standard.yaml"):
			with open(os.path.join(codes, file), "r") as reader:
				alt_code = yaml.safe_load(reader)
				freq_funcs = ef.build_functions(alt_code)
				freqs = ef.calculate_frequencies(freq_funcs, 0.5)["amino"]
				alt_freqs = np.array(list(dict(sorted(freqs.items())).values()))
				results[file.split(".")[0]] = np.sum(stand_freqs - np.log(stand_freqs / alt_freqs))

	df = pd.DataFrame.from_dict(results, orient="index", columns=["Kullback-Leibler-entropy"])
	df.index.name = "Genetic code"
	df["ID"] = df.index.map(map_df["ID"])
	df["Abbreviation"] = df.index.map(map_df["Abbreviation"])
	df.to_csv("code_kl_entropies.csv", sep="\t")