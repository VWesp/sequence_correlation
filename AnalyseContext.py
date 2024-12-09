import os
import sys
import gzip
import pandas as pd
from Bio import SeqIO
import collections as col
import multiprocessing as mp
from functools import partial


def get_neighbors(path, output, type, step, prog, size, lock):
	id = os.path.basename(path).split("_DNA.fasta.gz")[0]
	neighbors = col.defaultdict(lambda: col.defaultdict(int))
	filter = None
	if(type=="codon"):
		filter = [f"{a}{b}{c}" for a in "ACGT" for b in "ACGT" for c in "ACGT"]
	elif(type=="amino"):
		filter = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
				  "A", "G", "P", "T", "V", "L", "R", "S"]

	for el1 in filter:
		for el2 in filter:
			neighbors[el1][el2] = 0

	with gzip.open(path, "rt") as handle:
		dna_seqio = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))
		for dna_id,dna_rec in dna_seqio.items():
			seq = str(dna_rec.seq)
			elements = None
			if(type=="codon"):
				trim_seq = str(seq[:len(seq)-(len(seq)%3)])
				elements = [trim_seq[i:i+3] for i in range(0, len(trim_seq), 3)]
			elif(type=="amino"):
				elements = seq

			for i,el in enumerate(elements):
				left_neighbor = elements[i-step] if i-step >= 0 else "Start"
				right_neighbor = elements[i+step] if i+step < len(elements) else "End"
				neighbors[el][left_neighbor] += 1
				neighbors[el][right_neighbor] += 1

	df = pd.DataFrame.from_dict(neighbors)
	df = df[filter].loc[["Start"]+filter+["End"],:].fillna(0)

	res_output = os.path.join(os.path.join(output, str(step)), id+".csv")
	df.to_csv(res_output, sep="\t")

	with lock:
		prog.value += 1

	print(f"\rProgress {step}: {prog.value/size*100:.2f}%", end="")


if __name__=="__main__":
	path,output,type,procs = sys.argv[1:5]

	file_paths = None
	if(type=="codon"):
		file_paths = [os.path.join(path, file)
					  for file in os.listdir(path)
					  if file.endswith("_DNA.fasta.gz")]
	elif(type=="amino"):
		file_paths = [os.path.join(path, file)
					  for file in os.listdir(path)
					  if file.endswith(".fasta.gz")
					  and not file.endswith("_DNA.fasta.gz")]

	for step in range(1, 10, 1):
		os.makedirs(os.path.join(output, str(step)), exist_ok=True)

		mp.freeze_support()
	    # set the multiprocessing manager
		manager = mp.Manager()
		# set the lock for multiple processes
		lock = manager.Lock()
		# set the multiprocessing variable to count the progress
		prog = manager.Value("d", 0)

		print(f"\rProgress {step}: {prog.value/len(file_paths)*100:.2f}%",
			  end="")
		with mp.Pool(processes=int(procs)) as pool:
			pool_map = partial(get_neighbors, output=output, type=type,
							   step=step, prog=prog, size=len(file_paths),
							   lock=lock)
			process = pool.map_async(pool_map, file_paths)
			pool.close()
			#print(process.get())
			pool.join()

		print()
