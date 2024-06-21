import os
import sys
import wget
import numpy as np
import pandas as pd
from urllib.error import HTTPError


def downloadSingleProteomes(sprot_ids, stats_pd, output):
    uniprot_url = "https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/"
    prot_url = uniprot_url + "reference_proteomes/"
    eu_kingdom = ["fungi", "human", "invertebrates", "mammals", "plants",
                  "rodents", "vertebrates"]
    for index,row in stats_pd[["Proteome_ID", "Tax_ID"]].iterrows():
        prot_id = row["Proteome_ID"]
        tax_id = str(row["Tax_ID"])
        clade = None
        id_found = False
        for cl_type,ids in sprot_ids.items():
            if(int(tax_id) in ids):
                clade = cl_type
                id_found = True
                break

        if(not id_found):
            print("Not curated: "+tax_id)
            continue

        kingdom = clade
        if(clade in eu_kingdom):
            kingdom = "eukaryota"
            clade = os.path.join(kingdom, clade)

        clade_output = os.path.join(output, os.path.join(clade, "data"))
        os.makedirs(clade_output, exist_ok=True)

        success = False
        tries = 0
        while(not success):
            try:
                org_url = prot_url + kingdom.capitalize() + "/" + prot_id + "/"

                # download protein data
                prot_file_url = prot_id + "_" + tax_id + ".fasta.gz"
                prot_output = os.path.join(clade_output, prot_file_url)
                print(org_url+prot_file_url)
                wget.download(org_url+prot_file_url, prot_output)
                print()

                # download DNA data
                dna_file_url = prot_id + "_" + tax_id + "_DNA.fasta.gz"
                dna_output = os.path.join(clade_output, dna_file_url)
                print(org_url+dna_file_url)
                wget.download(org_url+dna_file_url, dna_output)
                success = True
                print()
            except HTTPError as err:
                tries += 1
                print("Failed download tries: "+str(tries))
                if(tries == 5):
                    break


if __name__ == "__main__":
    path_to_sprot = sys.argv[1]
    path_to_stats = sys.argv[2]
    output = sys.argv[3]

    sprot_ids = {"archaea": None, "bacteria": None, "fungi": None, "human": None,
                 "invertebrates": None, "mammals": None, "plants": None,
                 "rodents": None, "vertebrates": None, "viruses": None}

    for clade in sprot_ids:
        clade_path = os.path.join(path_to_sprot, clade+"_sprot_ids.csv")
        clade_df = pd.read_csv(clade_path)
        sprot_ids[clade] = list(clade_df["Tax_ID"])

    stats_pd = pd.read_csv(path_to_stats, sep="\t", skiprows=15)

    downloadSingleProteomes(sprot_ids, stats_pd, output)
