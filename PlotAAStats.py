import os
import math
import argparse
import numpy as np
import pandas as pd
import scipy as sci
import skbio as skb
import sklearn as skl
import seaborn as sns
import mpl_axes_aligner
import pypalettes as pp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec

plt.style.use("ggplot")


# Pandas describe_function but with median and mad instead of mean and std
def describe_data(x):
	if(isinstance(x, pd.Series)):
		desc_df = pd.Series(index=["count", "median", "mad", "min", "25%", "75%", "max"])
		desc_df["count"] = len(x)
		desc_df["median"] = np.median(x)
		desc_df["mad"] = sci.stats.median_abs_deviation(x)
		desc_df["min"] = np.min(x)
		desc_df["25%"] = np.percentile(x, 25)
		desc_df["75%"] = np.percentile(x, 75)
		desc_df["max"] = np.max(x)
		return desc_df.to_frame(name=np.nan).T
	else:
		stats = {col: [len(data), np.median(data), sci.stats.median_abs_deviation(data), np.min(data),
				  	   np.percentile(data, 25), np.percentile(data, 75), np.max(data)]
				 for col,data in x.items()}
		desc_df = pd.DataFrame(stats, index=["count", "median", "mad", "min", "25%", "75%", "max"])
		return desc_df


# Median absolute deviation interval
def mad_interval(x):
	median = np.median(x)
	mad = sci.stats.median_abs_deviation(x)
	return (median-mad, median+mad)


### Fisher's Z-transformation for averaging correlation coefficients
def fisher_Z_transform(x):
	z = [0.5*np.log((1+r)/(1-r)) for r in x]
	mean_z = np.mean(z)
	mean_r = (np.exp(2*mean_z)-1) / (np.exp(2*mean_z)+1)
	return mean_r


# Fisher's method for combining p-values
def fisher_method(x):
	chi_stat = -2 * np.sum(np.log(x))
	dof = 2 * len(x)
	comb_p = 1 - sci.stats.chi2.cdf(chi_stat, dof)
	return comb_p


# main method
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Plot proteome distribution statistics")
	parser.add_argument("-i", "--input", help="Specify the path to folder with the domains", required=True)
	parser.add_argument("-o", "--output", help="Set the path to the output folder", required=True)
	parser.add_argument("-r", "--resamples", help="Specify the number of resamples for the permutation tests (default: 9999)", type=int, default=9999)
	args = parser.parse_args()
	
	input = args.input
	output = args.output
	resamples = args.resamples
	
	os.makedirs(output, exist_ok=True)

	# Canonical amino acids order
	domains = ["Archaea", "Bacteria", "Eukaryota", "Viruses"]
	amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I", "A", "G", "P", "T", "V", "L", "R", "S"]
	aa_groups = {"Aliphatic": ["A", "G", "I", "L", "M", "V"], "Aromatic": ["F", "W", "Y"], "Charged": ["D", "E", "H", "K", "R"], "Uncharged": ["C", "N", "P", "Q", "S", "T"]}
	aa_group_order = [aa for group,aas in aa_groups.items() for aa in aas]
	domain_colors = ["red", "green", "blue", "orange"]
	corr_colors = pp.load_cmap("Acadia", keep_first_n=3).colors

	### Observed
	# Dataframe of observed frequencies
	frames = []
	for domain in domains:
		path = os.path.join(input, os.path.join(domain, os.path.join("output", "obs_frequencies.csv")))
		df = pd.read_csv(path, sep="\t", header=0, index_col=0)
		df["Domain"] = [domain] * len(df)
		frames.append(df)

	obs_freq_df = pd.concat(frames)
	# Dataframe of observed CLR frequencies
	frames = []
	for domain in domains:
		path = os.path.join(input, os.path.join(domain, os.path.join("output", "obs_clr.csv")))
		df = pd.read_csv(path, sep="\t", header=0, index_col=0)
		df["Domain"] = [domain] * len(df)
		frames.append(df)

	obs_clr_df = pd.concat(frames)
	### Code
	# Dataframe of code frequencies
	frames = []
	for domain in domains:
		path = os.path.join(input, os.path.join(domain, os.path.join("output", "code_frequencies.csv")))
		df = pd.read_csv(path, sep="\t", header=0, index_col=0)
		df["Domain"] = [domain] * len(df)
		frames.append(df)

	code_freq_df = pd.concat(frames)
	# Dataframe of code CLR frequencies
	frames = []
	for domain in domains:
		path = os.path.join(input, os.path.join(domain, os.path.join("output", "code_clr.csv")))
		df = pd.read_csv(path, sep="\t", header=0, index_col=0)
		df["Domain"] = [domain] * len(df)
		frames.append(df)

	code_clr_df = pd.concat(frames)
	# Dataframe of code CLR distances
	frames = []
	for domain in domains:
		path = os.path.join(input, os.path.join(domain, os.path.join("output", "code_clr_delta.csv")))
		df = pd.read_csv(path, sep="\t", header=0, index_col=0)
		df["Domain"] = [domain] * len(df)
		frames.append(df)

	code_clr_delta_df = pd.concat(frames)
	# Dataframe of code correlations
	frames = []
	for domain in domains:
		path = os.path.join(input, os.path.join(domain, os.path.join("output", "code_corrs.csv")))
		df = pd.read_csv(path, sep="\t", header=0, index_col=0)
		df["Domain"] = [domain] * len(df)
		frames.append(df)

	code_corr_df = pd.concat(frames)

	### Code+GC content
	# Dataframe of code+GC content frequencies
	frames = []
	for domain in domains:
		path = os.path.join(input, os.path.join(domain, os.path.join("output", "gc_frequencies.csv")))
		df = pd.read_csv(path, sep="\t", header=0, index_col=0)
		df["Domain"] = [domain] * len(df)
		frames.append(df)

	gc_freq_df = pd.concat(frames)
	# Dataframe of code+GC content CLR frequencies
	frames = []
	for domain in domains:
		path = os.path.join(input, os.path.join(domain, os.path.join("output", "gc_clr.csv")))
		df = pd.read_csv(path, sep="\t", header=0, index_col=0)
		df["Domain"] = [domain] * len(df)
		frames.append(df)

	gc_clr_df = pd.concat(frames)
	# Dataframe of code+GC content CLR distances
	frames = []
	for domain in domains:
		path = os.path.join(input, os.path.join(domain, os.path.join("output", "gc_clr_delta.csv")))
		df = pd.read_csv(path, sep="\t", header=0, index_col=0)
		df["Domain"] = [domain] * len(df)
		frames.append(df)

	gc_clr_delta_df = pd.concat(frames)
	# Dataframe of code+GC content correlations
	frames = []
	for domain in domains:
		path = os.path.join(input, os.path.join(domain, os.path.join("output", "gc_corrs.csv")))
		df = pd.read_csv(path, sep="\t", header=0, index_col=0)
		df["Domain"] = [domain] * len(df)
		frames.append(df)

	gc_corr_df = pd.concat(frames)

	###### Plot number of proteins
	g = sns.histplot(data=obs_freq_df, x="#Proteins", hue="Domain", alpha=0.5, kde=True, line_kws={"linewidth": 2, "linestyle": "--"}, stat="density", common_norm=False, 
				 	 log_scale=True, palette=domain_colors)
	g.set_xlabel("#Proteins", fontweight="bold", fontsize=10)
	g.set_ylabel("Density", fontweight="bold", fontsize=10)
	g.xaxis.grid(True, linestyle="--")
	g.legend(g.get_legend().legend_handles, domains, shadow=True)
	for ext in ["svg", "pdf"]:
		plt.savefig(os.path.join(output, f"number_of_proteins.{ext}"), bbox_inches="tight")
		
	plt.close()
	df_descr = obs_freq_df.groupby("Domain")["#Proteins"].apply(lambda x: describe_data(x)).reset_index(level=1, drop=True)
	df_descr.to_csv(os.path.join(output, "number_of_protein_stats.csv"), sep="\t")

	###### Plot median protein lengths
	g = sns.histplot(data=obs_freq_df, x="Length", hue="Domain", log_scale=10, alpha=0.5, kde=True, line_kws={"linewidth": 2, "linestyle": "--"}, stat="density",
				 	 common_norm=False, palette=domain_colors)
	g.set_xlabel("Length", fontweight="bold", fontsize=10)
	g.set_ylabel("Density", fontweight="bold", fontsize=10)
	g.xaxis.grid(True, linestyle="--")
	g.legend(g.get_legend().legend_handles, domains, shadow=True)
	for ext in ["svg", "pdf"]:
		plt.savefig(os.path.join(output, f"protein_lengths.{ext}"), bbox_inches="tight")
				
	plt.close()
	df_descr = obs_freq_df.groupby("Domain")["Length"].apply(lambda x: describe_data(x)).reset_index(level=1, drop=True)
	df_descr.to_csv(os.path.join(output, "protein_length_stats.csv"), sep="\t")

	###### Plot median gene GC contents
	g = sns.histplot(data=obs_freq_df, x="GC", hue="Domain", alpha=0.5, kde=True, line_kws={"linewidth": 2, "linestyle": "--"}, stat="density", common_norm=False, 
				 	 palette=domain_colors)
	g.set_xlabel("GC content", fontweight="bold", fontsize=10)
	g.set_ylabel("Density", fontweight="bold", fontsize=10)
	g.xaxis.grid(True, linestyle="--")
	g.legend(g.get_legend().legend_handles, domains, shadow=True)
	for ext in ["svg", "pdf"]:
		plt.savefig(os.path.join(output, f"gene_gc.{ext}"), bbox_inches="tight")
		
	plt.close()
	df_descr = obs_freq_df.groupby("Domain")["GC"].apply(lambda x: describe_data(x)).reset_index(level=1, drop=True)
	df_descr.to_csv(os.path.join(output, "gene_gc_stats.csv"), sep="\t")

	###### Plot median amino acid frequencies
	fig,axes = plt.subplots(3, 1, sharey=True)
	### Observed
	melted_df = obs_freq_df.melt(id_vars="Domain", value_vars=amino_acids, var_name="AminoAcid", value_name="medVal")
	sns.barplot(data=melted_df, x="AminoAcid", y="medVal", hue="Domain", estimator=np.median, errorbar=mad_interval, err_kws={"linewidth": 1.5}, palette=domain_colors, ax=axes[0])
	axes[0].set_xticks(np.arange(len(amino_acids)), amino_acids)
	axes[0].set_title(f"a) Based on distributions in proteomes", fontweight="bold", fontsize=10)
	axes[0].set_xlabel("")
	axes[0].set_ylabel("Frequency", fontweight="bold", fontsize=8)
	axes[0].xaxis.grid(True, linestyle="--")
	axes[0].legend([], frameon=False)
	frames = []
	for domain in domains:
		df_descr = describe_data(obs_freq_df[obs_freq_df["Domain"]==domain][amino_acids])
		df_descr.loc["Domain"] = domain
		df_descr = pd.concat([df_descr, pd.DataFrame([{}], index=[""])])
		frames.append(df_descr)

	pd.concat(frames).to_csv(os.path.join(output, "obs_freq_stats.csv"), sep="\t")
	### Code
	melted_df = code_freq_df.melt(id_vars="Domain", value_vars=amino_acids, var_name="AminoAcid", value_name="medVal")
	sns.barplot(data=melted_df, x="AminoAcid", y="medVal", hue="Domain", estimator=np.median, errorbar=mad_interval, err_kws={"linewidth": 1.5}, palette=domain_colors, ax=axes[1])
	axes[1].set_xticks(np.arange(len(amino_acids)), amino_acids)
	axes[1].set_title(f"b) Based on codon numbers", fontweight="bold", fontsize=10)
	axes[1].set_xlabel("")
	axes[1].set_ylabel("Frequency", fontweight="bold", fontsize=8)
	axes[1].xaxis.grid(True, linestyle="--")
	sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1, 1.1), shadow=True, title="")
	frames = []
	for domain in domains:
		df_descr = describe_data(code_freq_df[code_freq_df["Domain"]==domain][amino_acids])
		df_descr.loc["Domain"] = domain
		df_descr = pd.concat([df_descr, pd.DataFrame([{}], index=[""])])
		frames.append(df_descr)

	pd.concat(frames).to_csv(os.path.join(output, "code_freq_stats.csv"), sep="\t")
	### Code+GC content
	melted_df = gc_freq_df.melt(id_vars="Domain", value_vars=amino_acids, var_name="AminoAcid", value_name="medVal")
	sns.barplot(data=melted_df, x="AminoAcid", y="medVal", hue="Domain", estimator=np.median, errorbar=mad_interval, err_kws={"linewidth": 1.5}, palette=domain_colors, ax=axes[2])
	axes[2].set_xticks(np.arange(len(amino_acids)), amino_acids)
	axes[2].set_title(f"c) Based on codon numbers and GC contents", fontweight="bold", fontsize=10)
	axes[2].set_xlabel("Amino acid", fontweight="bold", fontsize=8)
	axes[2].set_ylabel("Frequency", fontweight="bold", fontsize=8)
	axes[2].xaxis.grid(True, linestyle="--")
	axes[2].legend([], frameon=False)
	frames = []
	for domain in domains:
		df_descr = describe_data(gc_freq_df[gc_freq_df["Domain"]==domain][amino_acids])
		df_descr.loc["Domain"] = domain
		df_descr = pd.concat([df_descr, pd.DataFrame([{}], index=[""])])
		frames.append(df_descr)

	pd.concat(frames).to_csv(os.path.join(output, "gc_freq_stats.csv"), sep="\t")
	###
	fig.subplots_adjust(hspace=0.7)
	for ext in ["svg", "pdf"]:
		plt.savefig(os.path.join(output, f"amino_acid_freqs.{ext}"), bbox_inches="tight")
		
	plt.close()

	###### Plot delta CLR as boxplots
	fig,axes = plt.subplots(2, 1, sharey=True)
	### Observed vs. code
	melted_df = code_clr_delta_df.melt(id_vars="Domain", value_vars=aa_group_order, var_name="AminoAcid", value_name="dCLR")
	axes[0].axhline(y=0, color="brown", linestyle="--", linewidth=1)
	sns.boxplot(data=melted_df, x="AminoAcid", y="dCLR", hue="Domain", showfliers=False, palette=domain_colors, ax=axes[0])
	axes[0].set_xticks(np.arange(len(aa_group_order)), aa_group_order)
	axes[0].set_title("a) Based on codon numbers", fontweight="bold", fontsize=10)
	axes[0].set_xlabel("")
	axes[0].set_ylabel("ΔCLR", fontweight="bold", fontsize=8)
	axes[0].xaxis.grid(True, linestyle="--")
	axes[0].legend([], frameon=False)
	group_pos = 0
	for group,aas in aa_groups.items():
		group_pos += len(aas)
		if(group_pos < 20):
			axes[0].axvline(x=group_pos-0.5, color="brown", linestyle="--", linewidth=1)

	frames = []
	for domain in domains:
		df_descr = describe_data(code_clr_delta_df[code_clr_delta_df["Domain"]==domain][aa_group_order])
		df_descr.loc["Domain"] = domain
		df_descr = pd.concat([df_descr, pd.DataFrame([{}], index=[""])])
		frames.append(df_descr)

	pd.concat(frames).to_csv(os.path.join(output, "code_dclr_stats.csv"), sep="\t")
	### Observed vs. code+GC content
	melted_df = gc_clr_delta_df.melt(id_vars="Domain", value_vars=aa_group_order, var_name="AminoAcid", value_name="dCLR")
	axes[1].axhline(y=0, color="brown", linestyle="--", linewidth=1)
	sns.boxplot(data=melted_df, x="AminoAcid", y="dCLR", hue="Domain", showfliers=False, palette=domain_colors, ax=axes[1])
	axes[1].set_xticks(np.arange(len(aa_group_order)), aa_group_order)
	axes[1].set_title("b) Based on codon numbers and GC contents", fontweight="bold", fontsize=10)
	axes[1].set_xlabel("Amino acid", fontweight="bold", fontsize=8)
	axes[1].set_ylabel("ΔCLR", fontweight="bold", fontsize=8)
	axes[1].xaxis.grid(True, linestyle="--")
	sns.move_legend(axes[1], "upper left", bbox_to_anchor=(0, 1.48), ncols=4, shadow=True, title="")
	group_pos = 0
	for group,aas in aa_groups.items():
		group_pos += len(aas)
		if(group_pos < 20):
			axes[1].axvline(x=group_pos-0.5, color="brown", linestyle="--", linewidth=1)

	frames = []
	for domain in domains:
		df_descr = describe_data(gc_clr_delta_df[gc_clr_delta_df["Domain"]==domain][aa_group_order])
		df_descr.loc["Domain"] = domain
		df_descr = pd.concat([df_descr, pd.DataFrame([{}], index=[""])])
		frames.append(df_descr)

	pd.concat(frames).to_csv(os.path.join(output, "gc_dclr_stats.csv"), sep="\t")
	###
	fig.subplots_adjust(hspace=0.7)
	for ext in ["svg", "pdf"]:
		plt.savefig(os.path.join(output, f"dclr_boxplot.{ext}"), bbox_inches="tight")
		
	plt.close()

	###### Plot delta CLR as radar plots
	fig,axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"})
	### Observed vs. code
	median_code_clr_delta_df = code_clr_delta_df.groupby("Domain")[aa_group_order].median()
	num_aas = len(median_code_clr_delta_df.columns)
	angles = [n / float(num_aas)*2*np.pi for n in range(num_aas)]
	angles += angles[:1]

	rmin = np.min(median_code_clr_delta_df.values) * 1.1
	rmax = np.max(median_code_clr_delta_df.values) * 1.1
	line_pos = 0
	index = 0
	for i,angle in enumerate(angles[:-1]):
	    if(i == line_pos): 
	        axes[0].plot([angle-0.15, angle-0.15], [rmin, rmax], color="brown", linestyle="dotted", linewidth=2, alpha=0.5)
	        line_pos += len(list(aa_groups.values())[index])
	        index += 1

	axes[0].plot(np.linspace(0, 2*np.pi, 100), [0]*100, color="brown", linestyle="dotted", linewidth=2, alpha=0.5)
	for i,(domain,row) in enumerate(median_code_clr_delta_df.iterrows()):
		aa_medians = row.to_list()
		aa_medians += aa_medians[:1]
		axes[0].plot(angles, aa_medians, label=domain, linewidth=2.5, linestyle="dashed", color=domain_colors[i], alpha=0.75)

	axes[0].set_xlabel("Amino acid", fontweight="bold", fontsize=12)
	axes[0].set_xticks(angles[:-1], aa_group_order)
	axes[0].set_ylabel(r"$\overline{\mathrm{ΔCLR}}$", fontweight="bold", labelpad=20, fontsize=12)
	axes[0].set_title("a) Based on codon numbers", fontweight="bold", pad=30, fontsize=14)
	axes[0].legend(bbox_to_anchor=(1.24, 1), shadow=True, fontsize=12, title="")
	### Observed vs. code+GC content
	median_gc_clr_delta_df = gc_clr_delta_df.groupby("Domain")[aa_group_order].median()
	num_aas = len(median_gc_clr_delta_df.columns)
	angles = [n / float(num_aas)*2*np.pi for n in range(num_aas)]
	angles += angles[:1]

	rmin = np.min(median_gc_clr_delta_df.values) * 1.1
	rmax = np.max(median_gc_clr_delta_df.values) * 1.1
	line_pos = 0
	index = 0
	for i,angle in enumerate(angles[:-1]):
	    if(i == line_pos): 
	        axes[1].plot([angle-0.15, angle-0.15], [rmin, rmax], color="brown", linestyle="dotted", linewidth=2, alpha=0.5)
	        line_pos += len(list(aa_groups.values())[index])
	        index += 1

	axes[1].plot(np.linspace(0, 2*np.pi, 100), [0]*100, color="brown", linestyle="dotted", linewidth=2, alpha=0.5)
	for i,(domain,row) in enumerate(median_gc_clr_delta_df.iterrows()):
		aa_medians = row.to_list()
		aa_medians += aa_medians[:1]
		axes[1].plot(angles, aa_medians, label=domain, linewidth=2.5, linestyle="dashed", color=domain_colors[i], alpha=0.75)

	axes[1].set_xlabel("Amino acid", fontweight="bold", fontsize=12)
	axes[1].set_xticks(angles[:-1], aa_group_order)
	axes[1].set_ylabel(r"$\overline{\mathrm{ΔCLR}}$", fontweight="bold", labelpad=20, fontsize=12)
	axes[1].set_title("b) Based on codon numbers and GC contents", fontweight="bold", pad=30, fontsize=14)
	###
	fig.set_size_inches(16, 10)
	for ext in ["svg", "pdf"]:
		plt.savefig(os.path.join(output, f"dclr_radar.{ext}"), bbox_inches="tight")
		
	plt.close()

	###### Plot Aitchison distances
	fig,axes = plt.subplots(2, 1, sharex=True, sharey=True)
	### Observed vs. code
	aitch_code_df = code_clr_delta_df[code_clr_delta_df["Aitchison_distance"]<=code_clr_delta_df["Aitchison_distance"].quantile(0.95)]
	sns.kdeplot(data=aitch_code_df, x="Aitchison_distance", hue="Domain", fill=True, common_norm=False, alpha=0.5, palette=domain_colors, ax=axes[0])
	axes[0].set_title(f"a) Based on codon numbers", fontweight="bold", fontsize=10)
	axes[0].set_xlabel("Aitchison distance", fontweight="bold", fontsize=8)
	axes[0].set_ylabel("Density", fontweight="bold", fontsize=8)
	axes[0].xaxis.set_tick_params(labelbottom=True)
	axes[0].xaxis.grid(True, linestyle="--")
	axes[0].legend([], frameon=False)
	#
	df_descr = code_clr_delta_df.groupby("Domain")["Aitchison_distance"].apply(lambda x: describe_data(x)).reset_index(level=1, drop=True)
	df_descr.to_csv(os.path.join(output, "code_aitchison_stats.csv"), sep="\t")
	### Observed vs. code+GC content
	aitch_gc_df = gc_clr_delta_df[gc_clr_delta_df["Aitchison_distance"]<=gc_clr_delta_df["Aitchison_distance"].quantile(0.95)]
	sns.kdeplot(data=aitch_gc_df, x="Aitchison_distance", hue="Domain", fill=True, common_norm=False, alpha=0.5, palette=domain_colors, ax=axes[1])
	axes[1].set_title(f"b) Based on codon numbers and GC contents", fontweight="bold", fontsize=10)
	axes[1].set_xlabel("Aitchison distance", fontweight="bold", fontsize=8)
	axes[1].set_ylabel("Density", fontweight="bold", fontsize=8)
	axes[1].xaxis.grid(True, linestyle="--")
	sns.move_legend(axes[1], "upper left", bbox_to_anchor=(0, 1.48), ncols=4, shadow=True, title="")
	#
	df_descr = gc_clr_delta_df.groupby("Domain")["Aitchison_distance"].apply(lambda x: describe_data(x)).reset_index(level=1, drop=True)
	df_descr.to_csv(os.path.join(output, "gc_aitchison_stats.csv"), sep="\t")
	###
	fig.subplots_adjust(hspace=0.7)
	for ext in ["svg", "pdf"]:
		plt.savefig(os.path.join(output, f"aitchison_distance.{ext}"), bbox_inches="tight")
		
	plt.close()

	###### Plot correlation coefficients
	fig,axes = plt.subplots(2, 2, sharey=True)
	i = 0
	j = 0
	label_lst = ["a)", "b)", "c)", "d)"]
	for index,domain in enumerate(domains):
		domain_code_df = code_corr_df[code_corr_df["Domain"]==domain]
		domain_code_corr_df = pd.DataFrame({"Pearson's ${r}$": domain_code_df["Pearson"], "Spearman's ${ρ}$": domain_code_df["Spearman"], "Kendall's ${τ}$": domain_code_df["Kendall"]}).melt(var_name="CorrelationTest", value_name="CorrelationCoefficient")
		domain_code_corr_df["Type"] = "Codon number"
		domain_gc_df = gc_corr_df[gc_corr_df["Domain"]==domain]
		domain_gc_corr_df = pd.DataFrame({"Pearson's ${r}$": domain_gc_df["Pearson"], "Spearman's ${ρ}$": domain_gc_df["Spearman"], "Kendall's ${τ}$": domain_gc_df["Kendall"]}).melt(var_name="CorrelationTest", value_name="CorrelationCoefficient")
		domain_gc_corr_df["Type"] = "Codon number+GC"
		comb_df = pd.concat([domain_code_corr_df, domain_gc_corr_df], ignore_index=True)
		sns.violinplot(data=comb_df, x="Type", y="CorrelationCoefficient", hue="CorrelationTest", palette=corr_colors, ax=axes[i,j])
		axes[i,j].set_title(f"{label_lst[index]} {domain}", fontweight="bold", fontsize=10)
		axes[i,j].set_xlabel("")
		axes[i,j].set_ylabel("Corr. coefficient", fontweight="bold", fontsize=8)
		axes[i,j].tick_params(axis="x", which="major", labelsize=8)
		axes[i,j].yaxis.set_tick_params(labelleft=True)
		axes[i,j].xaxis.grid(True, linestyle="--")
		if(i == 1 and j == 1):
			sns.move_legend(axes[i,j], "upper left", bbox_to_anchor=(-1.1, 1.48), ncols=3, shadow=True, title="")
		else:
			axes[i,j].legend([], frameon=False)
			
		j += 1
		if(j > 1):
			i += 1
			j = 0
		
	
	###
	fig.subplots_adjust(hspace=0.7)
	for ext in ["svg", "pdf"]:
		plt.savefig(os.path.join(output, f"corr_coefficients.{ext}"), bbox_inches="tight")
		
	plt.close()
	###
	corr_cols = ["Pearson", "Spearman", "Kendall"]
	p_cols = ["Pearson_p", "Pearson_q", "Spearman_p", "Spearman_q", "Kendall_p", "Kendall_q"]
	#
	fisher_code_corr_df = pd.DataFrame(index=domains, columns=[corr_cols+p_cols])
	fisher_code_corr_df.loc[domains, corr_cols] = code_corr_df.groupby("Domain")[corr_cols].agg(fisher_Z_transform)
	fisher_code_corr_df.loc[domains, p_cols] = code_corr_df.groupby("Domain")[p_cols].agg(fisher_method)
	fisher_code_corr_df.to_csv(os.path.join(output, "code_corr_stats.csv"), sep="\t")
	#
	fisher_gc_corr_df = pd.DataFrame(index=domains, columns=[corr_cols+p_cols])
	fisher_gc_corr_df.loc[domains, corr_cols] = gc_corr_df.groupby("Domain")[corr_cols].agg(fisher_Z_transform)
	fisher_gc_corr_df.loc[domains, p_cols] = gc_corr_df.groupby("Domain")[p_cols].agg(fisher_method)
	fisher_gc_corr_df.to_csv(os.path.join(output, "gc_corr_stats.csv"), sep="\t")

	###### Plot median observed frequencies of charged amino acids
	fig,axes = plt.subplots(2, 2, sharey=True)
	i = 0
	j = 0
	label_lst = ["a)", "b)", "c)", "d)"]
	for index,domain in enumerate(domains):
		domain_df = obs_freq_df[obs_freq_df["Domain"]==domain]
		gc_group = ((domain_df["GC"] // 0.05) * 0.05).values
		pos_df = pd.DataFrame({"Positively charged": domain_df[["H", "K", "R"]].sum(axis=1)}).melt(var_name="AminoAcid", value_name="Frequency")
		pos_df["GC content"] = gc_group
		neg_df = pd.DataFrame({"Negatively charged": domain_df[["D", "E"]].sum(axis=1)}).melt(var_name="AminoAcid", value_name="Frequency")
		neg_df["GC content"] = gc_group
		comb_df = pd.concat([pos_df, neg_df], ignore_index=True)
		sns.lineplot(data=comb_df, x="GC content", y="Frequency", hue="AminoAcid", style="AminoAcid", estimator=np.median, errorbar=mad_interval, markers=True, palette=["royalblue", "firebrick"], ax=axes[i,j])
		axes[i,j].set_title(f"{label_lst[index]} {domain}", fontweight="bold", fontsize=10)
		axes[i,j].xaxis.grid(True, linestyle="--")
		if(i == 1 and j == 1):
			sns.move_legend(axes[i,j], "upper left", bbox_to_anchor=(-1, 1.48), ncols=2, shadow=True, title="")
		else:
			axes[i,j].legend([], frameon=False)

		j += 1
		if(j > 1):
			i += 1
			j = 0

	###
	fig.subplots_adjust(hspace=0.9)
	for ext in ["svg", "pdf"]:
		plt.savefig(os.path.join(output, f"charged_amino_acid_freqs.{ext}"), bbox_inches="tight")
		
	plt.close()

	###### PCA for raw and CLR-transformed observed data
	fig, axes = plt.subplots(1, 2)
	pca_comp = skl.decomposition.PCA(n_components=2)
	### PCA for raw observed data
	pca_obs = pca_comp.fit_transform(obs_freq_df[amino_acids])
	sns.kdeplot(x=pca_obs[:, 0], y=pca_obs[:, 1], fill=True, ax=axes[0], cmap="magma")
	axes[0].set_xlabel(f"PC1 ({pca_comp.explained_variance_ratio_[0]*100:.2f}%)", fontweight="bold", fontsize=12)
	axes[0].set_ylabel(f"PC2 ({pca_comp.explained_variance_ratio_[1]*100:.2f}%)", fontweight="bold", fontsize=12)
	axes[0].set_title("a) Raw observations", fontweight="bold", fontsize=14)
	### PCA for CLR-transformed observed data
	pca_obs_clr = pca_comp.fit_transform(obs_clr_df[amino_acids])
	sns.kdeplot(x=pca_obs_clr[:, 0], y=pca_obs_clr[:, 1], fill=True, ax=axes[1], cmap="magma")
	axes[1].set_xlabel(f"PC1 ({pca_comp.explained_variance_ratio_[0]*100:.2f}%)", fontweight="bold", fontsize=12)
	axes[1].set_ylabel(f"PC2 ({pca_comp.explained_variance_ratio_[1]*100:.2f}%)", fontweight="bold", fontsize=12)
	axes[1].set_title("b) CLR-transformed observations", fontweight="bold", fontsize=14)
	###
	fig.subplots_adjust(wspace=0.25)
	fig.set_size_inches(12, 8)
	for ext in ["svg", "pdf"]:
		plt.savefig(os.path.join(output, f"observed_pca.{ext}"), bbox_inches="tight")

	plt.close()

	###### Plot correlations for amino acid frequencies and amino acid costs for Escherichia coli
	ecoli_obs_clr = np.array(obs_clr_df.loc[83333][amino_acids], dtype=float)
	rng = np.random.default_rng()
	perms = np.array([rng.permutation(len(ecoli_obs_clr)) for _ in range(resamples)])
	ecoli_obs_perms = ecoli_obs_clr[perms]

	def p_value(obs, perms):
		return (np.sum(np.abs(perms) >= abs(obs)) + 1) / (len(perms) + 1)

	coeffs = []
	p_values = []
	ecoli_aa_gluc = [18, -1, 8, 0,-7, 0, 3, 5, 2, -6, -2, 7, -1, -2, -2, 6, -2, -9, 0, -2]
	ecoli_aa_gylc = [16, -2, 6, -2, -11, 4.33, 4.33, 1, 0, -10, 6.33, 3, -3, 4, -6, 4, -6, -15, -4, -4]
	ecoli_aa_acet= [17, 6, 8, -1, -2, 2.33, 7.67, 4, 1, -1, 0.33, 6, -1, -2, 3, 5, -2, -5, 5, -2]
	for syn_values in [ecoli_aa_gluc, ecoli_aa_gylc, ecoli_aa_acet]:
		# Pearson
		coeff = sci.stats.pearsonr(ecoli_obs_clr, syn_values).statistic
		pears_perms = np.array([sci.stats.pearsonr(perm, syn_values).statistic for perm in ecoli_obs_perms])
		p_v = p_value(coeff, pears_perms)
		coeffs.append(coeff)
		p_values.append(p_v)
		# Spearman
		coeff = sci.stats.spearmanr(ecoli_obs_clr, syn_values).statistic
		spearm_perms = np.array([sci.stats.spearmanr(perm, syn_values).statistic for perm in ecoli_obs_perms])
		p_v = p_value(coeff, spearm_perms)
		coeffs.append(coeff)
		p_values.append(p_v)
		# Kendall
		coeff = sci.stats.kendalltau(ecoli_obs_clr, syn_values).statistic
		kendallt_perms = np.array([sci.stats.kendalltau(perm, syn_values).statistic for perm in ecoli_obs_perms])
		p_v = p_value(coeff, kendallt_perms)
		coeffs.append(coeff)
		p_values.append(p_v)

	###
	ecoli_df = pd.DataFrame(columns=["Correlation coefficient", "P-value", "Synthesis", "Correlation test"])
	ecoli_df["Correlation coefficient"] = coeffs
	ecoli_df["P-value"] = p_values
	ecoli_df["Synthesis"] = ["Glucose"]*3 + ["Glycerol"]*3 + ["Acetate"]*3
	ecoli_df["Correlation test"] = ["Pearson's ${r}$", "Spearman's ${ρ}$", "Kendall's ${τ}$"] * 3
	###
	g = sns.barplot(data=ecoli_df, x="Synthesis", y="Correlation coefficient", hue="Correlation test", palette=corr_colors)
	g.legend(shadow=True)
	g.set_xlabel("Starting synthesis compound", fontweight="bold", fontsize=10)
	g.set_ylabel("Corr. coefficient", fontweight="bold", fontsize=10)
	g.xaxis.grid(True, linestyle="--")
	for ext in ["svg", "pdf"]:
		plt.savefig(os.path.join(output, f"ecoli_cost_corr_coefficients.{ext}"), bbox_inches="tight")
		
	plt.close()
	###
	ecoli_df["Correlation test"] = ["Pearson", "Spearman", "Kendall's tau"] * 3
	ecoli_df.to_csv(os.path.join(output, "ecoli_cost_corr_stats.csv"), sep="\t", index=False)

