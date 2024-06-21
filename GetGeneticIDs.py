import os
import sys
import pandas as pd
from Bio import Entrez

if __name__ == "__main__":
    path_to_data = sys.argv[1]
    output = sys.argv[2]
    aa_dis_df = pd.read_csv(path_to_data, sep="\t", index_col=0, dtype=str)

    Entrez.email = "valentin.wesp@uni-jena.de"
    ids = set(aa_dis_df["Genome_Tax_ID"])
    ids = ['UP000000437_7955']
    tax_ids = []
    genetic_ids = []
    missing_ids = []
    id_count = 0
    print("Finished IDs: {}/{}".format(id_count, len(ids)), end="")
    for id in ids:
        try:
            txid = int(id.split("_")[1])
            handle = Entrez.efetch(db="taxonomy", id=txid)
            results = Entrez.read(handle)
            handle.close()
            genetic_id = int(results[0]["GeneticCode"]["GCId"])
            tax_ids.append(id)
            genetic_ids.append(genetic_id)
        except:
            missing_ids.append(id)
            pass

        id_count += 1
        print("\rFinished IDs: {}/{}".format(id_count, len(ids)), end="")

    print()
    print(missing_ids)
    id_df = pd.DataFrame(columns=["Genome_Tax_ID", "Genetic_ID"])
    id_df["Genome_Tax_ID"] = tax_ids
    id_df["Genetic_ID"] = genetic_id
    id_df.to_csv(output, sep="\t")
