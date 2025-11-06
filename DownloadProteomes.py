import os
import re
import time
import tqdm
import argparse
from ftplib import FTP
from urllib.error import HTTPError


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Download proteomes from the UniProt Knowledgebase")
	parser.add_argument("-d", "--domain", help="Specify from which domain the proteomes should be downloaded from", choices=["Archaea", "Bacteria", "Eukaryota", "Viruses"],
						required=True)
	parser.add_argument("-o", "--output", help="Set the path to the output folder", required=True)
	parser.add_argument("-t", "--tries", help="Set the number of retries to download a proteome file (default: 3)", type=int, default=3)
	parser.add_argument("-w", "--wait", help="Set the number of seconds to wait between each retry (default: 10)", type=int, default=10)
	args = parser.parse_args()
	
	domain = args.domain
	output = args.output
	max_tries = args.tries
	sleep = args.wait

	uniprot_url = "ftp.uniprot.org"
	database_url = "/pub/databases/uniprot/knowledgebase/reference_proteomes/"
	ftp = FTP(uniprot_url)
	ftp.login(user="anonymous", passwd="valentin.wesp@uni-jena.de")
	file_output = os.path.join(os.path.join(output, domain), "data")
	os.makedirs(file_output, exist_ok=True)
	output_prot_files = set([file.split("_")[0] for file in os.listdir(file_output)])
	output_dna_files = set([file.split("_")[0]+"_dna" for file in os.listdir(file_output)])

	domain_url = f"{database_url}{domain}/"
	ftp.cwd(domain_url)
	ids = [item for item in ftp.nlst() if not "." in item]
	with open(os.path.join(os.path.join(output, domain), "output.log"), "w+") as log_file:
		for id in tqdm.tqdm(ids, desc=f"Downloading files for {domain}"):
			if(id in output_prot_files and id+"_dna" in output_dna_files):
				continue

			tries = 1
			while(True):
				try:
					proteome_url = f"{domain_url}{id}/"
					ftp.cwd(proteome_url)
					files = ftp.nlst()
					prot_filter = list(filter(re.compile(f"{id}_[0-9]+.fasta.gz").match, files))
					dna_filter = list(filter(re.compile(f"{id}_[0-9]+_DNA.fasta.gz").match, files))
					if(len(prot_filter) and len(dna_filter)):
						prot_file = f"{proteome_url}{prot_filter[0]}"
						prot_output = os.path.join(file_output, prot_filter[0])
						if(not os.path.exists(prot_output)):
							with open(prot_output, "wb") as writer:
								ftp.retrbinary(f"RETR {prot_file}", writer.write)
						
						dna_file = f"{proteome_url}{dna_filter[0]}"
						dna_output = os.path.join(file_output, dna_filter[0])
						if(not os.path.exists(dna_output)):
							with open(dna_output, "wb") as writer:
								ftp.retrbinary(f"RETR {dna_file}", writer.write)
						
						log_file.write(f"\n{id} -> success -> {tries}/{max_tries} tries")
						log_file.flush()
					else:
						log_file.write(f"\n{id} -> no success -> {tries}/{max_tries} tries -> files are missing")
						log_file.flush()
						
					break
				except Exception:
					tries += 1
					log_file.write(f"\n\t-> sleep for {sleep}s")
					log_file.flush()
					ftp.close()
					time.sleep(sleep)
					ftp = FTP(uniprot_url)
					ftp.login(user="anonymous", passwd="valentin.wesp@uni-jena.de")
					if(tries > max_tries):
						log_file.write(f"\n{id} -> no success -> {tries}/{max_tries} tries")
						log_file.flush()
						break
				
		log_file.write("\n\t-> done")
		log_file.flush()
