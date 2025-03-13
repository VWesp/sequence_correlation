import os
import sys
import gzip
import pandas as pd
from Bio import SeqIO
import Bio.SeqUtils as util
import collections as defcol
import multiprocessing as mp
from functools import partial


def get_aa_dis(id, prot_dna_dict, type, output, progress, size, lock):
    res_output = os.path.join(output, id+".csv")
    prot_path = prot_dna_dict[id][0]
    dna_path = prot_dna_dict[id][1]
    all_aa_count = defcol.defaultdict(lambda: defcol.defaultdict())
    with gzip.open(prot_path, "rt") as prot_handle:
        prot_seqio = SeqIO.to_dict(SeqIO.parse(prot_handle, "fasta"))
        with gzip.open(dna_path, "rt") as dna_handle:
            dna_seqio = SeqIO.to_dict(SeqIO.parse(dna_handle, "fasta"))
            dna_seqio = {id.split("|")[1]:rec for id,rec in dna_seqio.items()}
            for prot_id,prot_rec in prot_seqio.items():
                prot_seq_type,prot_id,prot_name = prot_rec.description.split("|")[:3]
                if(type == "tr" or prot_seq_type == "sp"):
                    if(prot_id in dna_seqio):
                        dna_rec = dna_seqio[prot_id]
                        prot_seq = str(prot_rec.seq)
                        prot_len = len(prot_rec.seq)
                        dna_seq = str(dna_rec.seq)
                        amino_acids = set(prot_seq)
                        all_aa_count[prot_id] = {aa:prot_seq.count(aa)/prot_len
                                                 for aa in amino_acids}
                        all_aa_count[prot_id]["Name"] = prot_name
                        all_aa_count[prot_id]["Status"] = prot_seq_type
                        all_aa_count[prot_id]["GC"] = util.gc_fraction(dna_seq)
                        all_aa_count[prot_id]["Length"] = prot_len

    if(len(all_aa_count)):
        additional_cols = ["Name", "Status", "GC", "Length"]
        aa_codon_df = pd.DataFrame.from_dict(all_aa_count, orient="index")
        sorted_columns = sorted([col for col in aa_codon_df.columns
                                 if not col in additional_cols])
        aa_codon_df = aa_codon_df[additional_cols+sorted_columns]
        aa_codon_df.index.name = "Prot_ID"
        aa_codon_df.to_csv(res_output, sep="\t")

    with lock:
        progress.value += 1
        print("\r\033[2KProgress: {:.2f}% - last finished ID: {}".format(
                                           progress.value/size*100, id), end="")


# main method
if __name__ == "__main__":
    path_to_data = sys.argv[1]
    output = sys.argv[2]
    type = sys.argv[3]
    processes = int(sys.argv[4])

    os.makedirs(output, exist_ok=True)

    mp.freeze_support()
    # set the multiprocessing manager
    manager = mp.Manager()
    # set the lock for multiple processes
    lock = manager.Lock()
    # set the multiprocessing variable to count the progress
    progress = manager.Value("d", 0)

    prot_ids = [os.path.basename(file).split(".fasta.gz")[0]
                for file in os.listdir(path_to_data)
                if file.endswith(".fasta.gz")
                    and not file.endswith("_DNA.fasta.gz")]

    prot_dna_dict = manager.dict()
    for prot_id in prot_ids:
        prot_path = os.path.join(path_to_data, prot_id+".fasta.gz")
        dna_path = os.path.join(path_to_data, prot_id+"_DNA.fasta.gz")
        if(os.path.exists(prot_path) and os.path.exists(dna_path)):
            prot_dna_dict[prot_id] = [prot_path, dna_path]

    id_list = list(prot_dna_dict.keys())
    # start the multicore process for a given number of cores
    with mp.Pool(processes=processes) as pool:
        # run the process for the given parameters
        pool_map = partial(get_aa_dis, prot_dna_dict=prot_dna_dict, type=type,
                           output=output, progress=progress, size=len(id_list),
                           lock=lock)
        process = pool.map_async(pool_map, id_list)
        pool.close()
        pool.join()
        process.get()

    print()
