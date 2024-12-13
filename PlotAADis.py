import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
               "A", "G", "P", "T", "V", "L", "R", "S"]
prot_df = pd.read_csv(sys.argv[1], sep="\t", header=0, index_col=0)
kingdoms = ["Archaea", "Bacteria", "Eukaryota", "Viruses"]
aa_dis = {kingdom: prot_df.loc[f"{kingdom} frequency"]
          for kingdom in kingdoms}
aa_dis_df = pd.DataFrame(aa_dis)
aa_dis_std = {kingdom: prot_df.loc[f"{kingdom} frequency standard deviation"]
              for kingdom in kingdoms}
aa_dis_std_df = pd.DataFrame(aa_dis_std)
aa_dis_df.loc[amino_acids].plot(kind="bar", yerr=aa_dis_std_df.loc[amino_acids],
                                capsize=1, figsize=(14, 7), zorder=2)
plt.xlabel("Amino acids")
plt.ylabel("Mean distribution")
plt.title("Mean proteomic amino acid distribution across kingdoms")
plt.legend(title="Kingdom", loc="upper left")
plt.xticks(rotation=0)
plt.grid(alpha=0.5, zorder=0)
for ext in ["svg", "pdf"]:
    plt.savefig(f"{os.path.join(sys.argv[2], "mean_amino_acid_distribution")}.{ext}",
                bbox_inches="tight")

plt.close()
