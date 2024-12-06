import pandas as pd
import matplotlib.pyplot as plt


prot_df = pd.read_excel("ave_aa_dis_standard_comp.xlsx", header=0, index_col=0)
kingdoms = ["Archaea", "Bacteria", "Eukaryota", "Viruses"]
aa_dis ={kingdom: prot_df.loc[kingdom] for kingdom in kingdoms}
aa_dis_df = pd.DataFrame(aa_dis)
aa_dis_df.plot(kind="bar", figsize=(14, 7), zorder=2)
plt.xlabel("Amino acids")
plt.ylabel("Mean distribution")
plt.title("Mean proteomic amino acid distribution across kingdoms")
plt.legend(title="Kingdom", loc="upper left")
plt.xticks(rotation=0)
plt.grid(alpha=0.5, zorder=0)
for ext in ["svg", "pdf"]:
    plt.savefig(f"cor_results/mean_amino_acid_distribution.{ext}",
                bbox_inches="tight")

plt.close()
