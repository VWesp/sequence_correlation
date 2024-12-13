import re
import os
import sys
import gzip
import json
import pandas as pd
from Bio import SeqIO
import collections as col
import multiprocessing as mp
from functools import partial


def get_repeats(path, output, type, prog, size, lock):
    id = os.path.basename(path).split(".fasta.gz")[0]
    aa_repeats = col.defaultdict(lambda: col.defaultdict(lambda: col.defaultdict(int)))
    with gzip.open(path, "rt") as handle:
        seqio = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))
        for seq_id,rec in seqio.items():
            if(type == "tr" or seq_id.split("|")[0] == type):
                seq = str(rec.seq)
                seq_id = seq_id.split("|")[1]
                amino_acids = set(seq)
                for aa in amino_acids:
                    match = rf"{aa}+"
                    repeats = re.finditer(match, seq)
                    for rep in repeats:
                        rep_len = rep.end() - rep.start()
                        aa_repeats[f"{seq_id}:{len(seq)}"][aa][rep_len] += 1

    with open(os.path.join(output, id+".json"), "w", encoding="utf-8") as writer:
        json.dump(aa_repeats, writer, ensure_ascii=False, indent=4)

    with lock:
        prog.value += 1

    print(f"\rProgress: {prog.value/size*100:.2f}%", end="")


if __name__=="__main__":
    path,output,type,procs = sys.argv[1:5]

    file_paths = [os.path.join(path, file) for file in os.listdir(path)
                  if file.endswith(".fasta.gz")
                  and not file.endswith("_DNA.fasta.gz")]

    os.makedirs(output, exist_ok=True)

    mp.freeze_support()
    # set the multiprocessing manager
    manager = mp.Manager()
    # set the lock for multiple processes
    lock = manager.Lock()
    # set the multiprocessing variable to count the progress
    prog = manager.Value("d", 0)

    print(f"\rProgress: {prog.value/len(file_paths)*100:.2f}%", end="")
    with mp.Pool(processes=int(procs)) as pool:
        pool_map = partial(get_repeats, output=output, type=type, prog=prog,
                           size=len(file_paths), lock=lock)
        process = pool.map_async(pool_map, file_paths)
        pool.close()
        #print(process.get())
        pool.join()

    print()
