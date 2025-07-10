import os
import re
import gzip
import time
import tqdm
import argparse
import pandas as pd
from Bio import SeqIO
import Bio.SeqUtils as util
import multiprocessing as mp
from functools import partial


def get_proteome_distribution(data, output):
	file_id,prot_seqio,gene_seqio = data
	gene_seqio = {gene_id.split("|")[1]:rec for gene_id,rec in gene_seqio.items()}
	aa_dis_dict = {}
	for _,prot_rec in prot_seqio.items():
		seq_type,prot_id,prot_name = prot_rec.description.split("|")[:3]
		try:
			gene_rec = gene_seqio[prot_id]
			prot_seq = str(prot_rec.seq)
			prot_len = len(prot_seq)
			aa_dis_dict[prot_id] = {"Name": prot_name, "Status": seq_type, "Length": prot_len, "GC": util.gc_fraction(str(gene_rec.seq))}
			for aa in set(prot_seq):
				aa_dis_dict[prot_id][aa] = prot_seq.count(aa) / prot_len

		except KeyError as ke:
			continue
				
	if(len(aa_dis_dict)):
		aa_dis_df = pd.DataFrame.from_dict(aa_dis_dict, orient="index")
		aa_dis_df.index.name = "ProtID"
		aa_dis_df.to_csv(os.path.join(output, f"{file_id}.csv"), sep="\t")
		

# main method
if __name__ == "__main__":
	mp.freeze_support()
	
	parser = argparse.ArgumentParser(description="Compute amino acid distributions in proteomes")
	parser.add_argument("-d", "--data", help="Specify the path to the folder with the proteome files", required=True)
	parser.add_argument("-o", "--output", help="Set the path to the output folder", required=True)
	parser.add_argument("-c", "--chunks", help="Specify the chunk size; 0 loads all files at once (default: 100)", type=int, default=100)
	parser.add_argument("-t", "--threads", help="Specify the number of threads to be used (default: 1)", type=int, default=1)
	args = parser.parse_args()

	data_path = args.data
	output = args.output
	chunk_size = args.chunks
	threads = args.threads
	
	os.makedirs(output, exist_ok=True)
	
	prot_files = list(filter(re.compile(r"^(?!.*_DNA\.fasta\.gz$).*\.fasta\.gz$").match, os.listdir(data_path)))
	if(chunk_size <= 0):
		chunk_size = len(prot_files)
		
	for chunk in range(0, len(prot_files), chunk_size):
		seq_data = []
		chunked_files = prot_files[chunk:chunk+chunk_size]
		for file in tqdm.tqdm(chunked_files, desc=f"Loading proteome files for chunk [{chunk}-{min(chunk+chunk_size, len(prot_files))}]"):
			file_id = file.split(".fasta.gz")[0]
			with gzip.open(os.path.join(data_path, file), "rt") as prot_handle:
				prot_seqio = SeqIO.to_dict(SeqIO.parse(prot_handle, "fasta"))
				with gzip.open(os.path.join(data_path, f"{file_id}_DNA.fasta.gz"), "rt") as gene_handle:
					gene_seqio = SeqIO.to_dict(SeqIO.parse(gene_handle, "fasta"))
					seq_data.append([file_id, prot_seqio, gene_seqio])
					
		time.sleep(300)
		fdsfdsgfsd
		# start the multicore process for a given number of cores
		with mp.Pool(processes=threads) as pool:
			pool_map = partial(get_proteome_distribution, output=output)
			tqdm.tqdm(pool.imap(pool_map, seq_data), total=len(seq_data), desc=f"Calculating amino acid distributions for chunk [{chunk}-{min(chunk+chunk_size, len(prot_files))}]")

