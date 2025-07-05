import os
import sys
import tqdm
import argparse
import pandas as pd
from Bio import Entrez
import xml.etree.ElementTree as ET
from urllib.error import HTTPError


# main method
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Compute amino acid distributions in proteomes")
	parser.add_argument("-d", "--data", help="Specify the path to the folder with the distribution files", required=True)
	parser.add_argument("-o", "--output", help="Set the path to the output folder", required=True)
	args = parser.parse_args()
	
	data_path = args.data
	output = args.output
	
	tax_ids = [file.split(".csv")[0].split("_")[1] for file in os.listdir(data_path)]
	unfetched_ids = []
	encoding_df = pd.DataFrame(columns=["GeneticID", "MitoID", "PlastidID"], index=tax_ids)
	encoding_df.index.name = "TaxID"
	Entrez.email = "valentin.wesp@uni-jena.de"
	for tax_id in tqdm.tqdm(tax_ids, desc="Submitted IDs"):
		try:
			handle = Entrez.efetch(db="taxonomy", id=str(tax_id), retmode="xml")
			root = ET.fromstring(handle.read())
			genetic_id = root.find(".//GCId")
			if(genetic_id is not None):
				encoding_df.loc[tax_id, "GeneticID"] = genetic_id.text

			mito_id = root.find(".//MGCId")
			if(mito_id is not None):
				encoding_df.loc[tax_id, "MitoID"] = mito_id.text

			plastid_id = root.find(".//PropValueInt")
			if(plastid_id is not None):
				encoding_df.loc[tax_id, "PlastidID"] = plastid_id.text

			handle.close()
		except:
			continue

	encoding_df.replace("nan", "", inplace=True)
	encoding_df.to_csv(os.path.join(output, "encoding_data.csv"), sep="\t")

	with open(os.path.join(output, "unfetched_ids.csv"), "w") as writer:
		writer.write("TaxID\n"+"\n".join(unfetched_ids))
