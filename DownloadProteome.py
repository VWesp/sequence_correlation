import os
import sys
import time
import traceback
from ftplib import FTP
from urllib.error import HTTPError


if __name__ == "__main__":
    output = sys.argv[1]
    max_tries = int(sys.argv[2])
    sleep = int(sys.argv[3])

    uniprot_url = "ftp.uniprot.org"
    database_url = "/pub/databases/uniprot/knowledgebase/reference_proteomes/"
    print(f"Connecting to server: {uniprot_url}")
    ftp = FTP(uniprot_url)
    ftp.login(user="anonymous", passwd="valentin.wesp@uni-jena.de")
    kingdoms = ["Archaea", "Bacteria", "Eukaryota", "Viruses"]
    for kingdom in kingdoms:
        print(f"\tDownloading files for kingdom: {kingdom}")
        file_output = os.path.join(os.path.join(output, kingdom), "data")
        os.makedirs(file_output, exist_ok=True)
        output_prot_files = set([file.split("_")[0] for file in os.listdir(file_output)])
        output_dna_files = set([file.split("_")[0]+"_dna" for file in os.listdir(file_output)])

        kingdom_url = f"{database_url}{kingdom}/"
        ftp.cwd(kingdom_url)
        ids = [item for item in ftp.nlst() if not "." in item]
        count = 1
        for id in ids:
            print(f"\r\t\tDownloading files of proteome: {id} -> Progress: {count}/{len(ids)}", end="")
            count += 1
            if(id in output_prot_files and id+"_dna" in output_dna_files):
                continue

            tries = 0
            success = False
            while(not success):
                try:
                    proteome_url = f"{kingdom_url}{id}/"
                    ftp.cwd(proteome_url)
                    files = ftp.nlst()
                    data = [file for file in files if ".fasta.gz" in file]
                    if(len(data) >= 2):
                        prot_file = f"{proteome_url}{data[0]}"
                        prot_output = os.path.join(file_output, data[0])
                        if(not os.path.exists(prot_output)):
                            with open(prot_output, "wb") as writer:
                                ftp.retrbinary(f"RETR {prot_file}", writer.write)

                        dna_file = f"{proteome_url}{data[1]}"
                        dna_output = os.path.join(file_output, data[1])
                        if(not os.path.exists(dna_output)):
                            with open(dna_output, "wb") as writer:
                                ftp.retrbinary(f"RETR {dna_file}", writer.write)
                    else:
                        print()
                        print("\t\t\tFiles are missing. Ignoring proteome.")

                    success = True
                except Exception:
                    tries += 1
                    print()
                    print(traceback.format_exc())
                    print(f"\t\t\tDownload failed. Trying again after sleeping {sleep}s. -> Tries: {tries}/{max_tries}")
                    ftp.close()
                    if(tries == max_tries):
                        break

                    time.sleep(sleep)

                    ftp = FTP(uniprot_url)
                    ftp.login(user="anonymous", passwd="valentin.wesp@uni-jena.de")
                    ftp.cwd(kingdom_url)

        print()
