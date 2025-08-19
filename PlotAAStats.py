import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import pypalettes as pp
import scipy.stats as sci
import matplotlib.pyplot as plt

plt.style.use("ggplot")


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
	
	###### Plot median protein lengths
	if(not no_plot):
		g = sns.histplot(data=all_stats_df, x="Length_median", hue="Domain", log_scale=10, alpha=0.5, kde=True, line_kws={"linewidth": 2, "linestyle": "--"}, stat="density",
					 	 common_norm=False, palette=domain_colors)
		g.set_xlabel("Length", fontweight="bold", fontsize=10)
		g.set_ylabel("Density", fontweight="bold", fontsize=10)
		g.xaxis.grid(True, linestyle="--")
		g.legend(g.get_legend().legend_handles, domains, shadow=True)
		for ext in ["svg", "pdf"]:
			plt.savefig(os.path.join(output, f"median_protein_lengths.{ext}"), bbox_inches="tight")
					
		plt.close()
	
	###### Plot median gene GC contents
	if(not no_plot):
		g = sns.histplot(data=all_stats_df, x="GC_median", hue="Domain", alpha=0.5, kde=True, line_kws={"linewidth": 2, "linestyle": "--"}, stat="density", common_norm=False, 
					 	 palette=domain_colors)
		g.set_xlabel("GC content", fontweight="bold", fontsize=10)
		g.set_ylabel("Density", fontweight="bold", fontsize=10)
		g.xaxis.grid(True, linestyle="--")
		g.legend(g.get_legend().legend_handles, domains, shadow=True)
		for ext in ["svg", "pdf"]:
			plt.savefig(os.path.join(output, f"median_gene_gc.{ext}"), bbox_inches="tight")
			
		plt.close()
	
	###### Plot median amino acid frequencies
	if(not no_plot):
		fig,axes = plt.subplots(3, 1, sharey=True)
		### Empirical values
		aa_median_cols = [f"{aa}_median" for aa in amino_acids]
		melted_df = all_stats_df.melt(id_vars="Domain", value_vars=aa_median_cols, var_name="AminoAcid", value_name="MedianVal")
		sns.barplot(data=melted_df, x="AminoAcid", y="MedianVal", hue="Domain", errorbar=("pi", 50), err_kws={"linewidth": 1.5}, palette=domain_colors, ax=axes[0])
		axes[0].set_xticks(np.arange(len(amino_acids)), amino_acids)
		axes[0].set_title(f"a) Based on median distributions in protein", fontweight="bold", fontsize=10)
		axes[0].set_xlabel("")
		axes[0].set_ylabel("Frequency", fontweight="bold", fontsize=8)
		axes[0].xaxis.grid(True, linestyle="--")
		axes[0].legend([], frameon=False)
		### Values based on code
		aa_code_cols = [f"{aa}_code" for aa in amino_acids]
		melted_df = all_stats_df.melt(id_vars="Domain", value_vars=aa_code_cols, var_name="AminoAcid", value_name="MedianVal")
		sns.barplot(data=melted_df, x="AminoAcid", y="MedianVal", hue="Domain", errorbar=("pi", 50), err_kws={"linewidth": 1.5}, palette=domain_colors, ax=axes[1])
		axes[1].set_xticks(np.arange(len(amino_acids)), amino_acids)
		axes[1].set_title(f"b) Based on codon numbers", fontweight="bold", fontsize=10)
		axes[1].set_xlabel("")
		axes[1].set_ylabel("Frequency", fontweight="bold", fontsize=8)
		axes[1].xaxis.grid(True, linestyle="--")
		sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1, 1.2), shadow=True)
		### Values based on code and GC content
		aa_gc_cols = [f"{aa}_gc" for aa in amino_acids]
		melted_df = all_stats_df.melt(id_vars="Domain", value_vars=aa_gc_cols, var_name="AminoAcid", value_name="MedianVal")
		sns.barplot(data=melted_df, x="AminoAcid", y="MedianVal", hue="Domain", errorbar=("pi", 50), err_kws={"linewidth": 1.5}, palette=domain_colors, ax=axes[2])
		axes[2].set_xticks(np.arange(len(amino_acids)), amino_acids)
		axes[2].set_title(f"c) Based on codon numbers and GC contents", fontweight="bold", fontsize=10)
		axes[2].set_xlabel("Amino acid", fontweight="bold", fontsize=8)
		axes[2].set_ylabel("Frequency", fontweight="bold", fontsize=8)
		axes[2].xaxis.grid(True, linestyle="--")
		axes[2].legend([], frameon=False)
		###
		fig.subplots_adjust(hspace=0.7)
		for ext in ["svg", "pdf"]:
			plt.savefig(os.path.join(output, f"median_aa_freqs.{ext}"), bbox_inches="tight")
			
		plt.close()
	
	###### Plot percentage differences
	if(not no_plot):
		aa_groups = {"Aliphatic": ["A", "G", "I", "L", "M", "V"], "Aromatic": ["F",
					 "W", "Y"], "Charged": ["D", "E", "H", "K", "R"],
					 "Uncharged": ["C", "N", "P", "Q", "S", "T"]}	   
		aa_group_order = [aa for group in aa_groups.values() for aa in group]
		fig,axes = plt.subplots(2, 1, sharey=True)
		### Based on genetic codes
		aa_code_pct_cols = [f"{aa}_code_pct" for aa in aa_group_order]
		melted_df = all_stats_df.melt(id_vars="Domain", value_vars=aa_code_pct_cols, var_name="AminoAcid", value_name="PctVal")
		sns.boxplot(data=melted_df, x="AminoAcid", y="PctVal", hue="Domain", showfliers=False, palette=domain_colors, ax=axes[0])
		axes[0].set_xticks(np.arange(len(aa_group_order)), aa_group_order)
		axes[0].set_title(f"a) Based on codon numbers", fontweight="bold", fontsize=10)
		axes[0].set_xlabel("")
		axes[0].set_ylabel("Perc. difference", fontweight="bold", fontsize=8)
		axes[0].xaxis.grid(True, linestyle="--")
		axes[0].legend([], frameon=False)
		group_pos = 0
		for group,aas in aa_groups.items():
			group_pos += len(aas)
			if(group_pos < 20):
				axes[0].axvline(x=group_pos-0.5, color="brown", linestyle="--", linewidth=2)
		
		### Based on genetic codes and GC contents
		aa_gc_pct_cols = [f"{aa}_gc_pct" for aa in aa_group_order]
		melted_df = all_stats_df.melt(id_vars="Domain", value_vars=aa_gc_pct_cols, var_name="AminoAcid", value_name="PctVal")
		sns.boxplot(data=melted_df, x="AminoAcid", y="PctVal", hue="Domain", showfliers=False, palette=domain_colors, ax=axes[1])
		axes[1].set_xticks(np.arange(len(aa_group_order)), aa_group_order)
		axes[1].set_title(f"b) Based on codon numbers and GC contents", fontweight="bold", fontsize=10)
		axes[1].set_xlabel("Amino acid", fontweight="bold", fontsize=8)
		axes[1].set_ylabel("Perc. difference", fontweight="bold", fontsize=8)
		axes[1].xaxis.grid(True, linestyle="--")
		sns.move_legend(axes[1], "upper left", bbox_to_anchor=(0, 1.62), ncols=4, shadow=True)
		group_pos = 0
		for group,aas in aa_groups.items():
			group_pos += len(aas)
			if(group_pos < 20):
				axes[1].axvline(x=group_pos-0.5, color="brown", linestyle="--", linewidth=2)
			
		###
		fig.subplots_adjust(hspace=0.8)
		for ext in ["svg", "pdf"]:
			plt.savefig(os.path.join(output, f"pct_values.{ext}"), bbox_inches="tight")
			
		plt.close()
		
	###### Plot log-fold changes
	if(not no_plot):
		aa_groups = {"Aliphatic": ["A", "G", "I", "L", "M", "V"], "Aromatic": ["F",
					 "W", "Y"], "Charged": ["D", "E", "H", "K", "R"],
					 "Uncharged": ["C", "N", "P", "Q", "S", "T"]}	   
		aa_group_order = [aa for group in aa_groups.values() for aa in group]
		fig,axes = plt.subplots(2, 1, sharey=True)
		### Based on genetic codes
		aa_code_lfc_cols = [f"{aa}_code_lfc" for aa in aa_group_order]
		melted_df = all_stats_df.melt(id_vars="Domain", value_vars=aa_code_lfc_cols, var_name="AminoAcid", value_name="lFcVal")
		sns.boxplot(data=melted_df, x="AminoAcid", y="lFcVal", hue="Domain", showfliers=False, palette=domain_colors, ax=axes[0])
		axes[0].set_xticks(np.arange(len(aa_group_order)), aa_group_order)
		axes[0].set_title(f"a) Based on codon numbers", fontweight="bold", fontsize=10)
		axes[0].set_xlabel("")
		axes[0].set_ylabel("log-Fold change", fontweight="bold", fontsize=8)
		axes[0].xaxis.grid(True, linestyle="--")
		axes[0].legend([], frameon=False)
		group_pos = 0
		for group,aas in aa_groups.items():
			group_pos += len(aas)
			if(group_pos < 20):
				axes[0].axvline(x=group_pos-0.5, color="brown", linestyle="--", linewidth=2)
		
		### Based on genetic codes and GC contents
		aa_gc_lfc_cols = [f"{aa}_gc_lfc" for aa in aa_group_order]
		melted_df = all_stats_df.melt(id_vars="Domain", value_vars=aa_gc_lfc_cols, var_name="AminoAcid", value_name="lFcVal")
		sns.boxplot(data=melted_df, x="AminoAcid", y="lFcVal", hue="Domain", showfliers=False, palette=domain_colors, ax=axes[1])
		axes[1].set_xticks(np.arange(len(aa_group_order)), aa_group_order)
		axes[1].set_title(f"b) Based on codon numbers and GC contents", fontweight="bold", fontsize=10)
		axes[1].set_xlabel("Amino acid", fontweight="bold", fontsize=8)
		axes[1].set_ylabel("log-Fold change", fontweight="bold", fontsize=8)
		axes[1].xaxis.grid(True, linestyle="--")
		sns.move_legend(axes[1], "upper left", bbox_to_anchor=(0, 1.62), ncols=4, shadow=True)
		group_pos = 0
		for group,aas in aa_groups.items():
			group_pos += len(aas)
			if(group_pos < 20):
				axes[1].axvline(x=group_pos-0.5, color="brown", linestyle="--", linewidth=2)
			
		###
		fig.subplots_adjust(hspace=0.8)
		for ext in ["svg", "pdf"]:
			plt.savefig(os.path.join(output, f"lfc_values.{ext}"), bbox_inches="tight")
			
		plt.close()
	
	###### Plot median empirical frequencies of charged amino acids
	if(not no_plot):
		fig,axes = plt.subplots(2, 2, sharey=True)
		i = 0
		j = 0
		label_lst = ["a)", "b)", "c)", "d)"]
		for index,domain in enumerate(domains):
			domain_df = all_stats_df[all_stats_df["Domain"]==domain]
			gc_group = ((domain_df["GC_median"] // 0.05) * 0.05).values
			pos_df = pd.DataFrame({"Positively charged": domain_df[["H_median", "K_median", "R_median"]].sum(axis=1)}).melt(var_name="Amino acid", value_name="Frequency")
			pos_df["GC content"] = gc_group
			neg_df = pd.DataFrame({"Negatively charged": domain_df[["D_median", "E_median"]].sum(axis=1)}).melt(var_name="Amino acid", value_name="Frequency")
			neg_df["GC content"] = gc_group
			combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
			sns.lineplot(data=combined_df, x="GC content", y="Frequency", hue="Amino acid", style="Amino acid", errorbar=("pi", 50), markers=True,
						 palette=["firebrick", "royalblue"], ax=axes[i,j])
			axes[i,j].set_title(f"{label_lst[index]} {domain}", fontweight="bold", fontsize=10)
			axes[i,j].xaxis.grid(True, linestyle="--")
			if(i == 1 and j == 1):
				sns.move_legend(axes[i,j], "upper left", bbox_to_anchor=(-1, 1.62), ncols=2, shadow=True)
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
		fig.subplots_adjust(hspace=0.8)
		for ext in ["svg", "pdf"]:
			plt.savefig(os.path.join(output, f"median_charged_aa_freqs.{ext}"), bbox_inches="tight")
			
		plt.close()
	
	###### Plot log-RMSE 
	if(not no_plot):
		fig,axes = plt.subplots(2, 1, sharex=True, sharey=True)
		### Between empirical values and values Based on codon numbers
		sns.kdeplot(data=all_stats_df, x="log_RMSE_code", hue="Domain", fill=True, common_norm=False, alpha=0.5, palette=domain_colors, ax=axes[0])
		axes[0].set_title(f"a) Based on codon numbers", fontweight="bold", fontsize=10)
		axes[0].set_xlabel("RMSE", fontweight="bold", fontsize=8)
		axes[0].set_ylabel("Density", fontweight="bold", fontsize=8)
		axes[0].xaxis.set_tick_params(labelbottom=True)
		axes[0].xaxis.grid(True, linestyle="--")
		axes[0].legend([], frameon=False)
		### Between empirical values and values Based on codon numbers and GC contents
		sns.kdeplot(data=all_stats_df, x="log_RMSE_gc", hue="Domain", fill=True, common_norm=False, alpha=0.5, palette=domain_colors, ax=axes[1])
		axes[1].set_title(f"b) Based on codon numbers and GC contents", fontweight="bold", fontsize=10)
		axes[1].set_xlabel("log-RMSE", fontweight="bold", fontsize=8)
		axes[1].set_ylabel("Density", fontweight="bold", fontsize=8)
		axes[1].xaxis.grid(True, linestyle="--")
		sns.move_legend(axes[1], "upper left", bbox_to_anchor=(0, 1.62), ncols=4, shadow=True)
		###
		fig.subplots_adjust(hspace=0.8)
		for ext in ["svg", "pdf"]:
			plt.savefig(os.path.join(output, f"log_rmse_values.{ext}"), bbox_inches="tight")
			
		plt.close()
	
	###### Plot correlation coefficients
	if(not no_plot):
		fig,axes = plt.subplots(2, 2, sharey=True)
		i = 0
		j = 0
		label_lst = ["a)", "b)", "c)", "d)"]
		for index,domain in enumerate(domains):
			domain_df = all_stats_df[all_stats_df["Domain"]==domain]
			code_df = pd.DataFrame({"Pearson": domain_df["Ps_code"], "Spearman": domain_df["Sm_code"], "Kendall's tau": domain_df["Kt_code"]}).melt(var_name="Correlation test", 
																																			   value_name="Correlation coefficient")
			code_df["Type"] = "Codon number"
			gc_df = pd.DataFrame({"Pearson": domain_df["Ps_gc"], "Spearman": domain_df["Sm_gc"], "Kendall's tau": domain_df["Kt_gc"]}).melt(var_name="Correlation test", 
																																			value_name="Correlation coefficient")
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
				sns.move_legend(axes[i,j], "upper left", bbox_to_anchor=(-1, 1.62), ncols=3, shadow=True)
			else:
				axes[i,j].legend([], frameon=False)
				
			j += 1
			if(j > 1):
				i += 1
				j = 0
			
		###
		fig.subplots_adjust(hspace=0.8)
		for ext in ["svg", "pdf"]:
			plt.savefig(os.path.join(output, f"corr_coefficients.{ext}"), bbox_inches="tight")
			
		plt.close()
	
	###### Plot correlations for amino acid frequencies and amino acid costs for Escherichia coli
	if(not no_plot):
		ecoli_df = pd.DataFrame(columns=["Correlation coefficient", "P-value", "Synthesis", "Correlation test"])
		aa_median_cols = [f"{aa}_median" for aa in amino_acids]
		ecoli_amino_acids = np.asarray(all_stats_df.loc[83333][aa_median_cols], dtype=float)
		### For glucose
		ecoli_aa_gloc = [18, -1, 8, 0,-7, 0, 3, 5, 2, -6, -2, 7, -1, -2, -2, 6, -2, -9, 0, -2]
		# Pearson
		ecoli_df.loc[0, "Correlation coefficient"] = sci.pearsonr(ecoli_amino_acids, ecoli_aa_gloc).statistic
		ecoli_df.loc[0, "P-value"]  = sci.permutation_test((ecoli_amino_acids,), lambda x: sci.pearsonr(x, ecoli_aa_gloc).statistic, permutation_type="pairings", 
															n_resamples=resamples).pvalue
		ecoli_df.loc[0, "Synthesis"] = "Glucose"
		ecoli_df.loc[0, "Correlation test"] = "Pearson"
		# Spearman
		ecoli_df.loc[1, "Correlation coefficient"]  = sci.spearmanr(ecoli_amino_acids, ecoli_aa_gloc).statistic
		ecoli_df.loc[1, "P-value"] = sci.permutation_test((ecoli_amino_acids,), lambda x: sci.spearmanr(x, ecoli_aa_gloc).statistic, permutation_type="pairings", 
														   n_resamples=resamples).pvalue
		ecoli_df.loc[1, "Synthesis"] = "Glucose"
		ecoli_df.loc[1, "Correlation test"] = "Spearman"
		# Kendall's tau
		ecoli_df.loc[2, "Correlation coefficient"] = sci.kendalltau(ecoli_amino_acids, ecoli_aa_gloc).statistic
		ecoli_df.loc[2, "P-value"] = sci.permutation_test((ecoli_amino_acids,), lambda x: sci.kendalltau(x, ecoli_aa_gloc).statistic, permutation_type="pairings", 
														   n_resamples=resamples).pvalue
		ecoli_df.loc[2, "Synthesis"] = "Glucose"
		ecoli_df.loc[2, "Correlation test"] = "Kendall's tau"
		### For glycerol
		ecoli_aa_gylc = [16, -2, 6, -2, -11, 4.33, 4.33, 1, 0, -10, 6.33, 3, -3, 4, -6, 4, -6, -15, -4, -4]
		# Pearson
		ecoli_df.loc[3, "Correlation coefficient"] = sci.pearsonr(ecoli_amino_acids, ecoli_aa_gylc).statistic
		ecoli_df.loc[3, "P-value"]  = sci.permutation_test((ecoli_amino_acids,), lambda x: sci.pearsonr(x, ecoli_aa_gylc).statistic, permutation_type="pairings", 
															n_resamples=resamples).pvalue
		ecoli_df.loc[3, "Synthesis"] = "Glycerol"
		ecoli_df.loc[3, "Correlation test"] = "Pearson"
		# Spearman
		ecoli_df.loc[4, "Correlation coefficient"]  = sci.spearmanr(ecoli_amino_acids, ecoli_aa_gylc).statistic
		ecoli_df.loc[4, "P-value"] = sci.permutation_test((ecoli_amino_acids,), lambda x: sci.spearmanr(x, ecoli_aa_gylc).statistic, permutation_type="pairings", 
														   n_resamples=resamples).pvalue
		ecoli_df.loc[4, "Synthesis"] = "Glycerol"
		ecoli_df.loc[4, "Correlation test"] = "Spearman"
		# Kendall's tau
		ecoli_df.loc[5, "Correlation coefficient"] = sci.kendalltau(ecoli_amino_acids, ecoli_aa_gylc).statistic
		ecoli_df.loc[5, "P-value"] = sci.permutation_test((ecoli_amino_acids,), lambda x: sci.kendalltau(x, ecoli_aa_gylc).statistic, permutation_type="pairings", 
														   n_resamples=resamples).pvalue
		ecoli_df.loc[5, "Synthesis"] = "Glycerol"
		ecoli_df.loc[5, "Correlation test"] = "Kendall's tau"
		### For acetate
		ecoli_aa_acet= [17, 6, 8, -1, -2, 2.33, 7.67, 4, 1, -1, 0.33, 6, -1, -2, 3, 5, -2, -5, 5, -2]
		# Pearson
		ecoli_df.loc[6, "Correlation coefficient"] = sci.pearsonr(ecoli_amino_acids, ecoli_aa_acet).statistic
		ecoli_df.loc[6, "P-value"]  = sci.permutation_test((ecoli_amino_acids,), lambda x: sci.pearsonr(x, ecoli_aa_acet).statistic, permutation_type="pairings", 
															n_resamples=resamples).pvalue
		ecoli_df.loc[6, "Synthesis"] = "Acetate"
		ecoli_df.loc[6, "Correlation test"] = "Pearson"
		# Spearman
		ecoli_df.loc[7, "Correlation coefficient"]  = sci.spearmanr(ecoli_amino_acids, ecoli_aa_acet).statistic
		ecoli_df.loc[7, "P-value"] = sci.permutation_test((ecoli_amino_acids,), lambda x: sci.spearmanr(x, ecoli_aa_acet).statistic, permutation_type="pairings", 
														   n_resamples=resamples).pvalue
		ecoli_df.loc[7, "Synthesis"] = "Acetate"
		ecoli_df.loc[7, "Correlation test"] = "Spearman"
		# Kendall's tau
		ecoli_df.loc[8, "Correlation coefficient"] = sci.kendalltau(ecoli_amino_acids, ecoli_aa_acet).statistic
		ecoli_df.loc[8, "P-value"] = sci.permutation_test((ecoli_amino_acids,), lambda x: sci.kendalltau(x, ecoli_aa_acet).statistic, permutation_type="pairings", 
														   n_resamples=resamples).pvalue
		ecoli_df.loc[8, "Synthesis"] = "Acetate"
		ecoli_df.loc[8, "Correlation test"] = "Kendall's tau"
		ecoli_df.to_csv(os.path.join(output, "ecoli_cost_corr_coefficients.csv"), sep="\t", index=False)
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

