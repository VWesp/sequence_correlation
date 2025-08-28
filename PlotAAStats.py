import os
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
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec

plt.style.use("ggplot")


### Fisher's Z-transformation
def fisher_Z(x):
	z = [0.5*np.log((1+r)/(1-r)) for r in x]
	mean_z = np.mean(z)
	mean_r = (np.exp(2*mean_z)-1) / (np.exp(2*mean_z)+1)
	return mean_r


# main method
if __name__ == "__main__":
	parser = argparse.ArgumentParser(
	description="Plot proteome distribution statistics")
	parser.add_argument("-i", "--input", help="Specify the path to folder with the domains", required=True)
	parser.add_argument("-o", "--output", help="Set the path to the output folder", required=True)
	parser.add_argument("-r", "--resamples", help="Specify the number of resamples for the permutation tests (default: 9999)", type=int, default=9999)
	parser.add_argument("-np", "--no-plot", help="Specify to not plot everything", action="store_true")
	args = parser.parse_args()
	
	input = args.input
	output = args.output
	resamples = args.resamples
	no_plot = args.no_plot
	
	os.makedirs(output, exist_ok=True)
	
	# Canonical amino acids order
	amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I", "A", "G", "P", "T", "V", "L", "R", "S"]
	aa_groups = {"Aliphatic": ["A", "G", "I", "L", "M", "V"], "Aromatic": ["F", "W", "Y"], "Charged": ["D", "E", "H", "K", "R"], "Uncharged": ["C", "N", "P", "Q", "S", "T"]}	   
	aa_group_order = [aa for group in aa_groups.values() for aa in group]
	aa_code_cols = [f"{aa}_code" for aa in amino_acids]
	aa_gc_cols = [f"{aa}_gc" for aa in amino_acids]
	aa_code_delta_clr = [f"{aa}_code_clr_delta" for aa in aa_group_order]
	aa_gc_delta_clr = [f"{aa}_gc_clr_delta" for aa in aa_group_order]
	
	domains = ["Archaea", "Bacteria", "Eukaryota", "Viruses"]
	all_stats = []
	for domain in domains:
		domain_path = os.path.join(input, domain)
		stats_df = pd.read_csv(os.path.join(domain_path, "combined_distributions.csv"), sep="\t", header=0, index_col=0)
		stats_df["Domain"] = [domain] * len(stats_df)
		all_stats.append(stats_df)
	
	all_stats_df = pd.concat(all_stats)
	
	domain_colors = ["red", "green", "blue", "orange"]
	corr_colors = pp.load_cmap("Acadia", keep_first_n=3).colors
	
	###### Plot number of proteins
	if(not no_plot):
		g = sns.histplot(data=all_stats_df, x="#Proteins", hue="Domain", alpha=0.5, kde=True, line_kws={"linewidth": 2, "linestyle": "--"}, stat="density", common_norm=False, 
					 	 log_scale=True, palette=domain_colors)
		g.set_xlabel("#Proteins", fontweight="bold", fontsize=10)
		g.set_ylabel("Density", fontweight="bold", fontsize=10)
		g.xaxis.grid(True, linestyle="--")
		g.legend(g.get_legend().legend_handles, domains, shadow=True)
		for ext in ["svg", "pdf"]:
			plt.savefig(os.path.join(output, f"number_of_proteins.{ext}"), bbox_inches="tight")
			
		plt.close()
		
		d_series = []
		for domain in domains:
			domain_df = all_stats_df[all_stats_df["Domain"]==domain]
			data = domain_df["#Proteins"].describe()
			data.loc["Sum"] = domain_df["#Proteins"].sum()
			data.rename(domain, inplace=True)
			d_series.append(data)
			
		pd.concat(d_series, axis=1).to_csv(os.path.join(output, "number_of_proteins.csv"), sep="\t")
	
	###### Plot mean protein lengths
	if(not no_plot):
		g = sns.histplot(data=all_stats_df, x="Length", hue="Domain", log_scale=10, alpha=0.5, kde=True, line_kws={"linewidth": 2, "linestyle": "--"}, stat="density",
					 	 common_norm=False, palette=domain_colors)
		g.set_xlabel("Length", fontweight="bold", fontsize=10)
		g.set_ylabel("Density", fontweight="bold", fontsize=10)
		g.xaxis.grid(True, linestyle="--")
		g.legend(g.get_legend().legend_handles, domains, shadow=True)
		for ext in ["svg", "pdf"]:
			plt.savefig(os.path.join(output, f"mean_protein_lengths.{ext}"), bbox_inches="tight")
					
		plt.close()
		
		d_series = []
		for domain in domains:
			domain_df = all_stats_df[all_stats_df["Domain"]==domain]
			data = domain_df["Length"].describe()
			data.rename(domain, inplace=True)
			d_series.append(data)
			
		pd.concat(d_series, axis=1).to_csv(os.path.join(output, "mean_protein_lengths.csv"), sep="\t")
	
	###### Plot mean gene GC contents
	if(not no_plot):
		g = sns.histplot(data=all_stats_df, x="GC", hue="Domain", alpha=0.5, kde=True, line_kws={"linewidth": 2, "linestyle": "--"}, stat="density", common_norm=False, 
					 	 palette=domain_colors)
		g.set_xlabel("GC content", fontweight="bold", fontsize=10)
		g.set_ylabel("Density", fontweight="bold", fontsize=10)
		g.xaxis.grid(True, linestyle="--")
		g.legend(g.get_legend().legend_handles, domains, shadow=True)
		for ext in ["svg", "pdf"]:
			plt.savefig(os.path.join(output, f"mean_gene_gc.{ext}"), bbox_inches="tight")
			
		plt.close()
		
		d_series = []
		for domain in domains:
			domain_df = all_stats_df[all_stats_df["Domain"]==domain]
			data = domain_df["GC"].describe()
			data.rename(domain, inplace=True)
			d_series.append(data)
			
		pd.concat(d_series, axis=1).to_csv(os.path.join(output, "mean_gene_gc.csv"), sep="\t")
	
	###### Plot mean amino acid frequencies
	if(not no_plot):
		fig,axes = plt.subplots(3, 1, sharey=True)
		### Empirical values
		melted_df = all_stats_df.melt(id_vars="Domain", value_vars=amino_acids, var_name="AminoAcid", value_name="meanVal")
		sns.barplot(data=melted_df, x="AminoAcid", y="meanVal", hue="Domain", errorbar="sd", err_kws={"linewidth": 1.5}, palette=domain_colors, ax=axes[0])
		axes[0].set_xticks(np.arange(len(amino_acids)), amino_acids)
		axes[0].set_title(f"a) Based on mean distributions in proteomes", fontweight="bold", fontsize=10)
		axes[0].set_xlabel("")
		axes[0].set_ylabel("Frequency", fontweight="bold", fontsize=8)
		axes[0].xaxis.grid(True, linestyle="--")
		axes[0].legend([], frameon=False)
		d_series = []
		for domain in domains:
			domain_df = all_stats_df[all_stats_df["Domain"]==domain]
			data = domain_df[amino_acids].describe()
			data.loc["Domain",:] = domain
			data.loc["",:] = ""
			d_series.append(data)
		
		d_df = pd.concat(d_series)
		d_df.columns = amino_acids
		d_df.to_csv(os.path.join(output, "mean_obs_freqs.csv"), sep="\t")
		### Values based on code
		melted_df = all_stats_df.melt(id_vars="Domain", value_vars=aa_code_cols, var_name="AminoAcid", value_name="meanVal")
		sns.barplot(data=melted_df, x="AminoAcid", y="meanVal", hue="Domain", errorbar="sd", err_kws={"linewidth": 1.5}, palette=domain_colors, ax=axes[1])
		axes[1].set_xticks(np.arange(len(amino_acids)), amino_acids)
		axes[1].set_title(f"b) Based on codon numbers", fontweight="bold", fontsize=10)
		axes[1].set_xlabel("")
		axes[1].set_ylabel("Frequency", fontweight="bold", fontsize=8)
		axes[1].xaxis.grid(True, linestyle="--")
		sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1, 1.1), shadow=True, title="")
		d_series = []
		for domain in domains:
			domain_df = all_stats_df[all_stats_df["Domain"]==domain]
			data = domain_df[aa_code_cols].describe()
			data.loc["Domain",:] = domain
			data.loc["",:] = ""
			d_series.append(data)
			
		d_df = pd.concat(d_series)
		d_df.columns = amino_acids
		d_df.to_csv(os.path.join(output, "mean_code_freqs.csv"), sep="\t")
		### Values based on code and GC content
		melted_df = all_stats_df.melt(id_vars="Domain", value_vars=aa_gc_cols, var_name="AminoAcid", value_name="meanVal")
		sns.barplot(data=melted_df, x="AminoAcid", y="meanVal", hue="Domain", errorbar="sd", err_kws={"linewidth": 1.5}, palette=domain_colors, ax=axes[2])
		axes[2].set_xticks(np.arange(len(amino_acids)), amino_acids)
		axes[2].set_title(f"c) Based on codon numbers and GC contents", fontweight="bold", fontsize=10)
		axes[2].set_xlabel("Amino acid", fontweight="bold", fontsize=8)
		axes[2].set_ylabel("Frequency", fontweight="bold", fontsize=8)
		axes[2].xaxis.grid(True, linestyle="--")
		axes[2].legend([], frameon=False)
		d_series = []
		for domain in domains:
			domain_df = all_stats_df[all_stats_df["Domain"]==domain]
			data = domain_df[aa_gc_cols].describe()
			data.loc["Domain",:] = domain
			data.loc["",:] = ""
			d_series.append(data)
			
		d_df = pd.concat(d_series)
		d_df.columns = amino_acids
		d_df.to_csv(os.path.join(output, "mean_gc_freqs.csv"), sep="\t")
		###
		fig.subplots_adjust(hspace=0.7)
		for ext in ["svg", "pdf"]:
			plt.savefig(os.path.join(output, f"mean_aa_freqs.{ext}"), bbox_inches="tight")
			
		plt.close()
	
	###### Plot delta CLR
	if(not no_plot):
		fig,axes = plt.subplots(2, 1, sharey=True)
		### Based on genetic codes
		melted_df = all_stats_df.melt(id_vars="Domain", value_vars=aa_code_delta_clr, var_name="AminoAcid", value_name="dCLR")
		sns.boxplot(data=melted_df, x="AminoAcid", y="dCLR", hue="Domain", showfliers=False, palette=domain_colors, ax=axes[0])
		axes[0].set_xticks(np.arange(len(aa_group_order)), aa_group_order)
		axes[0].set_title(f"a) Based on codon numbers", fontweight="bold", fontsize=10)
		axes[0].set_xlabel("")
		axes[0].set_ylabel("ΔCLR", fontweight="bold", fontsize=8)
		axes[0].xaxis.grid(True, linestyle="--")
		axes[0].legend([], frameon=False)
		group_pos = 0
		for group,aas in aa_groups.items():
			group_pos += len(aas)
			if(group_pos < 20):
				axes[0].axvline(x=group_pos-0.5, color="brown", linestyle="--", linewidth=2)
				
		d_series = []
		for domain in domains:
			domain_df = all_stats_df[all_stats_df["Domain"]==domain]
			data = domain_df[aa_code_delta_clr].abs().describe()
			data.loc["Domain",:] = domain
			data.loc["",:] = ""
			d_series.append(data)
		
		d_df = pd.concat(d_series)
		d_df.columns = aa_group_order
		d_df.to_csv(os.path.join(output, "delta_code_clr.csv"), sep="\t")
		### Based on genetic codes and GC contents
		melted_df = all_stats_df.melt(id_vars="Domain", value_vars=aa_gc_delta_clr, var_name="AminoAcid", value_name="dCLR")
		sns.boxplot(data=melted_df, x="AminoAcid", y="dCLR", hue="Domain", showfliers=False, palette=domain_colors, ax=axes[1])
		axes[1].set_xticks(np.arange(len(aa_group_order)), aa_group_order)
		axes[1].set_title(f"b) Based on codon numbers and GC contents", fontweight="bold", fontsize=10)
		axes[1].set_xlabel("Amino acid", fontweight="bold", fontsize=8)
		axes[1].set_ylabel("ΔCLR", fontweight="bold", fontsize=8)
		axes[1].xaxis.grid(True, linestyle="--")
		sns.move_legend(axes[1], "upper left", bbox_to_anchor=(0, 1.48), ncols=4, shadow=True, title="")
		group_pos = 0
		for group,aas in aa_groups.items():
			group_pos += len(aas)
			if(group_pos < 20):
				axes[1].axvline(x=group_pos-0.5, color="brown", linestyle="--", linewidth=2)
		
		d_series = []
		for domain in domains:
			domain_df = all_stats_df[all_stats_df["Domain"]==domain]
			data = domain_df[aa_gc_delta_clr].abs().describe()
			data.loc["Domain",:] = domain
			data.loc["",:] = ""
			d_series.append(data)
		
		d_df = pd.concat(d_series)
		d_df.columns = aa_group_order
		d_df.to_csv(os.path.join(output, "delta_gc_clr.csv"), sep="\t")
		###
		fig.subplots_adjust(hspace=0.7)
		for ext in ["svg", "pdf"]:
			plt.savefig(os.path.join(output, f"dclr_values.{ext}"), bbox_inches="tight")
			
		plt.close()
	
	###### Plot mean empirical frequencies of charged amino acids
	if(not no_plot):
		fig,axes = plt.subplots(2, 2, sharey=True)
		i = 0
		j = 0
		label_lst = ["a)", "b)", "c)", "d)"]
		for index,domain in enumerate(domains):
			domain_df = all_stats_df[all_stats_df["Domain"]==domain]
			gc_group = ((domain_df["GC"] // 0.05) * 0.05).values
			pos_df = pd.DataFrame({"Positively charged": domain_df[["H", "K", "R"]].sum(axis=1)}).melt(var_name="AminoAcid", value_name="Frequency")
			pos_df["GC content"] = gc_group
			neg_df = pd.DataFrame({"Negatively charged": domain_df[["D", "E"]].sum(axis=1)}).melt(var_name="AminoAcid", value_name="Frequency")
			neg_df["GC content"] = gc_group
			combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
			sns.lineplot(data=combined_df, x="GC content", y="Frequency", hue="AminoAcid", style="AminoAcid", errorbar="sd", markers=True,
						 palette=["royalblue", "firebrick"], ax=axes[i,j])
			axes[i,j].set_title(f"{label_lst[index]} {domain}", fontweight="bold", fontsize=10)
			axes[i,j].xaxis.grid(True, linestyle="--")
			if(i == 1 and j == 1):
				sns.move_legend(axes[i,j], "upper left", bbox_to_anchor=(-1, 1.48), ncols=2, shadow=True, title="")
			else:
				axes[i,j].legend([], frameon=False)
			
			if(i == 1):
				axes[i,j].set_xlabel("GC content", fontweight="bold", fontsize=8)
			else:
				axes[i,j].set_xlabel("")
				
			if(j == 0):
				axes[i,j].set_ylabel("Frequency", fontweight="bold", fontsize=8)
			
			j += 1
			if(j > 1):
				i += 1
				j = 0
			
		###
		fig.subplots_adjust(hspace=0.7)
		for ext in ["svg", "pdf"]:
			plt.savefig(os.path.join(output, f"mean_charged_aa_freqs.{ext}"), bbox_inches="tight")
			
		plt.close()
	
	###### Plot Aitchison distances
	if(not no_plot):
		fig,axes = plt.subplots(2, 1, sharex=True, sharey=True)
		### Between empirical values and values Based on codon numbers
		aitch_code_df = all_stats_df[all_stats_df["aitchison_code"]<=all_stats_df["aitchison_code"].quantile(0.99)]
		sns.kdeplot(data=aitch_code_df, x="aitchison_code", hue="Domain", fill=True, common_norm=False, alpha=0.5, palette=domain_colors, ax=axes[0])
		axes[0].set_title(f"a) Based on codon numbers", fontweight="bold", fontsize=10)
		axes[0].set_xlabel("Aitchison distance", fontweight="bold", fontsize=8)
		axes[0].set_ylabel("Density", fontweight="bold", fontsize=8)
		axes[0].xaxis.set_tick_params(labelbottom=True)
		axes[0].xaxis.grid(True, linestyle="--")
		axes[0].legend([], frameon=False)
		d_series = []
		for domain in domains:
			domain_df = all_stats_df[all_stats_df["Domain"]==domain]
			data = domain_df["aitchison_code"].describe()
			data.rename(domain, inplace=True)
			d_series.append(data)
			
		pd.concat(d_series, axis=1).to_csv(os.path.join(output, "aitchison_code.csv"), sep="\t")
		### Between empirical values and values Based on codon numbers and GC contents
		aitch_gc_df = all_stats_df[all_stats_df["aitchison_gc"]<=all_stats_df["aitchison_gc"].quantile(0.99)]
		sns.kdeplot(data=aitch_gc_df, x="aitchison_gc", hue="Domain", fill=True, common_norm=False, alpha=0.5, palette=domain_colors, ax=axes[1])
		axes[1].set_title(f"b) Based on codon numbers and GC contents", fontweight="bold", fontsize=10)
		axes[1].set_xlabel("Aitchison distance", fontweight="bold", fontsize=8)
		axes[1].set_ylabel("Density", fontweight="bold", fontsize=8)
		axes[1].xaxis.grid(True, linestyle="--")
		sns.move_legend(axes[1], "upper left", bbox_to_anchor=(0, 1.48), ncols=4, shadow=True, title="")
		d_series = []
		for domain in domains:
			domain_df = all_stats_df[all_stats_df["Domain"]==domain]
			data = domain_df["aitchison_gc"].describe()
			data.rename(domain, inplace=True)
			d_series.append(data)
			
		pd.concat(d_series, axis=1).to_csv(os.path.join(output, "aitchison_gc.csv"), sep="\t")
		###
		fig.subplots_adjust(hspace=0.7)
		for ext in ["svg", "pdf"]:
			plt.savefig(os.path.join(output, f"aitchison_distance.{ext}"), bbox_inches="tight")
			
		plt.close()
	
	###### Plot Biplot
	if(not no_plot):
		for domain in domains:
			fig,axes = plt.subplots(2, 1, sharex=True, sharey=True)
			domain_df = all_stats_df[all_stats_df["Domain"]==domain]
			######
			z_clr = domain_df[aa_code_delta_clr] - domain_df[aa_code_delta_clr].mean()
			z_clr.columns = amino_acids
			pca = skl.decomposition.PCA()
			scores = pca.fit_transform(z_clr)
			df_scores = pd.DataFrame(scores, columns=[f"PC{i}" for i in range(1, z_clr.shape[1]+1)])
			df_loadings = pd.DataFrame(pca.components_, columns=z_clr.columns, index=df_scores.columns)
			axes[0].scatter(df_scores["PC1"].values, df_scores["PC2"].values, color="royalblue", edgecolor="black", alpha=0.5)
			axes[0].set_xlabel("PC1", fontweight="bold", fontsize=8)
			axes[0].set_ylabel("PC2", fontweight="bold", fontsize=8)
			ax = axes[0].twinx().twiny()
			for col in df_loadings.columns:
				tipx = df_loadings.loc["PC1", col]
				tipy = df_loadings.loc["PC2", col]
				ax.arrow(0, 0, tipx, tipy, color="firebrick", alpha=0.5)
				ax.text(tipx*1.05, tipy*1.05, col, fontdict={"color": "peru", "weight": "bold", "size": 6}, ha="center", va="center",
						 path_effects=[pe.withStroke(linewidth=4, foreground="black")])
				
			mpl_axes_aligner.align.xaxes(axes[0], 0, ax, 0, 0.5)
			mpl_axes_aligner.align.yaxes(axes[0], 0, ax, 0, 0.5)
			axes[0].set_title(f"a) Based on codon numbers", fontweight="bold", fontsize=10)
			axes[0].xaxis.set_tick_params(labelbottom=True)
			axes[0].xaxis.grid(True, linestyle="--")
			######
			z_clr = domain_df[aa_gc_delta_clr] - domain_df[aa_gc_delta_clr].mean()
			z_clr.columns = amino_acids
			pca = skl.decomposition.PCA()
			scores = pca.fit_transform(z_clr)
			df_scores = pd.DataFrame(scores, columns=[f"PC{i}" for i in range(1, z_clr.shape[1]+1)])
			df_loadings = pd.DataFrame(pca.components_, columns=z_clr.columns, index=df_scores.columns)
			axes[1].scatter(df_scores["PC1"].values, df_scores["PC2"].values, color="royalblue", edgecolor="black", alpha=0.5)
			axes[1].set_xlabel("PC1", fontweight="bold", fontsize=8)
			axes[1].set_ylabel("PC2", fontweight="bold", fontsize=8)
			ax = axes[1].twinx().twiny()
			for col in df_loadings.columns:
				tipx = df_loadings.loc["PC1", col]
				tipy = df_loadings.loc["PC2", col]
				ax.arrow(0, 0, tipx, tipy, color="firebrick", alpha=0.5)
				ax.text(tipx*1.05, tipy*1.05, col, fontdict={"color": "peru", "weight": "bold", "size": 6}, ha="center", va="center",
						 path_effects=[pe.withStroke(linewidth=4, foreground="black")])
				
			mpl_axes_aligner.align.xaxes(axes[1], 0, ax, 0, 0.5)
			mpl_axes_aligner.align.yaxes(axes[1], 0, ax, 0, 0.5)
			axes[1].set_title(f"b) Based on codon numbers and GC contents", fontweight="bold", fontsize=10)
			axes[1].xaxis.set_tick_params(labelbottom=True)
			axes[1].xaxis.grid(True, linestyle="--")
			######
			fig.subplots_adjust(hspace=0.6)
			for ext in ["svg", "pdf"]:
				plt.savefig(os.path.join(output, f"pca_dclr_{domain.lower()}.{ext}"), bbox_inches="tight")
				
			plt.close()
	
	###### Plot correlation coefficients
	if(not no_plot):
		fig,axes = plt.subplots(2, 2, sharey=True)
		i = 0
		j = 0
		label_lst = ["a)", "b)", "c)", "d)"]
		d_code_series = []
		d_gc_series = []
		for index,domain in enumerate(domains):
			domain_df = all_stats_df[all_stats_df["Domain"]==domain]
			code_df = pd.DataFrame({"Pearson ${r}$": domain_df["Ps_code"], "Spearman ${ρ}$": domain_df["Sm_code"], "Kendall's tau ${τ}$": domain_df["Kt_code"]}).melt(var_name="Correlation test", value_name="Correlation coefficient")
			code_df["Type"] = "Codon number"
			gc_df = pd.DataFrame({"Pearson ${r}$": domain_df["Ps_gc"], "Spearman ${ρ}$": domain_df["Sm_gc"], "Kendall's tau ${τ}$": domain_df["Kt_gc"]}).melt(var_name="Correlation test", value_name="Correlation coefficient")
			gc_df["Type"] = "Codon number+GC"
			comb_df = pd.concat([code_df, gc_df], ignore_index=True)
			sns.violinplot(data=comb_df, x="Type", y="Correlation coefficient", hue="Correlation test", palette=corr_colors, ax=axes[i,j])
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
			
			corr_df = pd.DataFrame(columns=["Pearson", "Pearson_p", "Spearman", "Spearman_p", "Kendall", "Kendall_p"])
			corr_df.loc[domain, "Pearson"] = fisher_Z(domain_df["Ps_code"])
			corr_df.loc[domain, "Pearson_p"] = fisher_Z(domain_df["Ps_code_p"])
			corr_df.loc[domain, "Spearman"] = fisher_Z(domain_df["Sm_code"])
			corr_df.loc[domain, "Spearman_p"] = fisher_Z(domain_df["Sm_code_p"])
			corr_df.loc[domain, "Kendall"] = fisher_Z(domain_df["Kt_code"])
			corr_df.loc[domain, "Kendall_p"] = fisher_Z(domain_df["Kt_code_p"])
			d_code_series.append(corr_df)
			
			corr_df = pd.DataFrame(columns=["Pearson", "Pearson_p", "Spearman", "Spearman_p", "Kendall", "Kendall_p"])
			corr_df.loc[domain, "Pearson"] = fisher_Z(domain_df["Ps_gc"])
			corr_df.loc[domain, "Pearson_p"] = fisher_Z(domain_df["Ps_gc_p"])
			corr_df.loc[domain, "Spearman"] = fisher_Z(domain_df["Sm_gc"])
			corr_df.loc[domain, "Spearman_p"] = fisher_Z(domain_df["Sm_gc_p"])
			corr_df.loc[domain, "Kendall"] = fisher_Z(domain_df["Kt_gc"])
			corr_df.loc[domain, "Kendall_p"] = fisher_Z(domain_df["Kt_gc_p"])
			d_gc_series.append(corr_df)
			
		pd.concat(d_code_series).to_csv(os.path.join(output, "code_corr_coefficient.csv"), sep="\t")
		pd.concat(d_gc_series).to_csv(os.path.join(output, "gc_corr_coefficient.csv"), sep="\t")
		###
		fig.subplots_adjust(hspace=0.7)
		for ext in ["svg", "pdf"]:
			plt.savefig(os.path.join(output, f"corr_coefficients.{ext}"), bbox_inches="tight")
			
		plt.close()
	
	###### Plot correlations for amino acid frequencies and amino acid costs for Escherichia coli
	if(not no_plot):
		ecoli_df = pd.DataFrame(columns=["Correlation coefficient", "P-value", "Synthesis", "Correlation test"])
		ecoli_amino_acids = np.asarray(all_stats_df.loc[83333][amino_acids], dtype=float)
		ecoli_clr = skb.stats.composition.clr(ecoli_amino_acids)
		### For glucose
		ecoli_aa_gloc = [18, -1, 8, 0,-7, 0, 3, 5, 2, -6, -2, 7, -1, -2, -2, 6, -2, -9, 0, -2]
		# Pearson
		corr_stats = sci.stats.permutation_test((ecoli_clr,), lambda x: sci.stats.pearsonr(x, ecoli_aa_gloc).statistic, permutation_type="pairings", n_resamples=resamples)
		ecoli_df.loc[0, "Correlation coefficient"] = corr_stats.statistic
		ecoli_df.loc[0, "P-value"]  = corr_stats.pvalue
		ecoli_df.loc[0, "Synthesis"] = "Glucose"
		ecoli_df.loc[0, "Correlation test"] = "Pearson ${r}$"
		# Spearman
		corr_stats = sci.stats.permutation_test((ecoli_clr,), lambda x: sci.stats.spearmanr(x, ecoli_aa_gloc).statistic, permutation_type="pairings", n_resamples=resamples)
		ecoli_df.loc[1, "Correlation coefficient"] = corr_stats.statistic
		ecoli_df.loc[1, "P-value"] = corr_stats.pvalue
		ecoli_df.loc[1, "Synthesis"] = "Glucose"
		ecoli_df.loc[1, "Correlation test"] = "Spearman ${ρ}$"
		# Kendall's tau
		corr_stats = sci.stats.permutation_test((ecoli_clr,), lambda x: sci.stats.kendalltau(x, ecoli_aa_gloc).statistic, permutation_type="pairings", n_resamples=resamples)
		ecoli_df.loc[2, "Correlation coefficient"] = corr_stats.statistic
		ecoli_df.loc[2, "P-value"] = corr_stats.pvalue
		ecoli_df.loc[2, "Synthesis"] = "Glucose"
		ecoli_df.loc[2, "Correlation test"] = "Kendall's tau ${τ}$"
		### For glycerol
		ecoli_aa_gylc = [16, -2, 6, -2, -11, 4.33, 4.33, 1, 0, -10, 6.33, 3, -3, 4, -6, 4, -6, -15, -4, -4]
		# Pearson
		corr_stats = sci.stats.permutation_test((ecoli_clr,), lambda x: sci.stats.pearsonr(x, ecoli_aa_gylc).statistic, permutation_type="pairings", n_resamples=resamples)
		ecoli_df.loc[3, "Correlation coefficient"] = corr_stats.statistic
		ecoli_df.loc[3, "P-value"]  = corr_stats.pvalue
		ecoli_df.loc[3, "Synthesis"] = "Glycerol"
		ecoli_df.loc[3, "Correlation test"] = "Pearson ${r}$"
		# Spearman
		corr_stats = sci.stats.permutation_test((ecoli_clr,), lambda x: sci.stats.spearmanr(x, ecoli_aa_gylc).statistic, permutation_type="pairings", n_resamples=resamples)
		ecoli_df.loc[4, "Correlation coefficient"]  = corr_stats.statistic
		ecoli_df.loc[4, "P-value"] = corr_stats.pvalue
		ecoli_df.loc[4, "Synthesis"] = "Glycerol"
		ecoli_df.loc[4, "Correlation test"] = "Spearman ${ρ}$"
		# Kendall's tau
		corr_stats = sci.stats.permutation_test((ecoli_clr,), lambda x: sci.stats.kendalltau(x, ecoli_aa_gylc).statistic, permutation_type="pairings", n_resamples=resamples)
		ecoli_df.loc[5, "Correlation coefficient"] = corr_stats.statistic
		ecoli_df.loc[5, "P-value"] = corr_stats.pvalue
		ecoli_df.loc[5, "Synthesis"] = "Glycerol"
		ecoli_df.loc[5, "Correlation test"] = "Kendall's tau ${τ}$"
		### For acetate
		ecoli_aa_acet= [17, 6, 8, -1, -2, 2.33, 7.67, 4, 1, -1, 0.33, 6, -1, -2, 3, 5, -2, -5, 5, -2]
		# Pearson
		corr_stats = sci.stats.permutation_test((ecoli_clr,), lambda x: sci.stats.pearsonr(x, ecoli_aa_acet).statistic, permutation_type="pairings", n_resamples=resamples)
		ecoli_df.loc[6, "Correlation coefficient"] = corr_stats.statistic
		ecoli_df.loc[6, "P-value"]  = corr_stats.pvalue
		ecoli_df.loc[6, "Synthesis"] = "Acetate"
		ecoli_df.loc[6, "Correlation test"] = "Pearson ${r}$"
		# Spearman
		corr_stats = sci.stats.permutation_test((ecoli_clr,), lambda x: sci.stats.spearmanr(x, ecoli_aa_acet).statistic, permutation_type="pairings",n_resamples=resamples)
		ecoli_df.loc[7, "Correlation coefficient"]  = corr_stats.statistic
		ecoli_df.loc[7, "P-value"] = corr_stats.pvalue
		ecoli_df.loc[7, "Synthesis"] = "Acetate"
		ecoli_df.loc[7, "Correlation test"] = "Spearman ${ρ}$"
		# Kendall's tau
		corr_stats = sci.stats.permutation_test((ecoli_clr,), lambda x: sci.stats.kendalltau(x, ecoli_aa_acet).statistic, permutation_type="pairings", n_resamples=resamples)
		ecoli_df.loc[8, "Correlation coefficient"] = corr_stats.statistic
		ecoli_df.loc[8, "P-value"] = corr_stats.pvalue
		ecoli_df.loc[8, "Synthesis"] = "Acetate"
		ecoli_df.loc[8, "Correlation test"] = "Kendall's tau ${τ}$"
		###
		g = sns.barplot(data=ecoli_df, x="Synthesis", y="Correlation coefficient", hue="Correlation test", palette=corr_colors)
		hatches = ["/", ".", "\\",]
		for i in range(len(hatches)):
			for bar in g.containers[i]:
				bar.set_hatch(hatches[i%len(hatches)])
				bar.set_edgecolor("silver")
				
		g.legend(shadow=True)
		g.set_xlabel("Starting synthesis compound", fontweight="bold", fontsize=10)
		g.set_ylabel("Corr. coefficient", fontweight="bold", fontsize=10)
		g.xaxis.grid(True, linestyle="--")
		for ext in ["svg", "pdf"]:
			plt.savefig(os.path.join(output, f"ecoli_cost_corr_coefficients.{ext}"), bbox_inches="tight")
			
		plt.close()
		###
		ecoli_df["Correlation test"] = ["Pearson", "Spearman", "Kendall's tau"] * 3
		ecoli_df.to_csv(os.path.join(output, "ecoli_cost_corr_coefficients.csv"), sep="\t", index=False)

