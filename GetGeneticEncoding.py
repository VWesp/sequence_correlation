import os
import sys
import numpy as np
import pandas as pd
from Bio import Entrez
import xml.etree.ElementTree as ET
from urllib.error import HTTPError


# main method
if __name__ == "__main__":
    input = sys.argv[1]
    output = sys.argv[2]

    tax_df = pd.read_csv(input, sep="\t", header=0)
    tax_ids = np.array(tax_df["TaxID"])

    unfetched_ids = []
    encoding_df = pd.DataFrame(columns=["GeneticID", "MitoID", "PlastidID"])
    encoding_df.index.name = "TaxID"
    Entrez.email = "valentin.wesp@uni-jena.de"
    for index,id in enumerate(tax_ids):
        sys.stdout.write("\033[2K")
        print(f"\rSubmitting {index+1}/{len(tax_ids)}: {id}", end="")
        try:
            handle = Entrez.efetch(db="taxonomy", id=str(id), retmode="xml")
            root = ET.fromstring(handle.read())
            genetic_id = root.find(".//GCId")
            if(genetic_id is not None):
                encoding_df.loc[id, "GeneticID"] = genetic_id.text

            mito_id = root.find(".//MGCId")
            if(mito_id is not None):
                encoding_df.loc[id, "MitoID"] = mito_id.text

            plastid_id = root.find(".//PropValueInt")
            if(plastid_id is not None):
                encoding_df.loc[id, "PlastidID"] = plastid_id.text

            handle.close()
        except:
            unfetched_ids.append(str(id))

    print()
    encoding_df.replace("nan", "", inplace=True)
    encoding_df.to_csv(os.path.join(output, "encoding_data.csv"), sep="\t")

    with open(os.path.join(output, "unfetched_ids.csv"), "w") as writer:
        writer.write("TaxID\n"+"\n".join(unfetched_ids))
