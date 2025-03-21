import io
import os
import sys
import gzip
import time
import paramiko
import numpy as np
import pandas as pd
from Bio import SeqIO
import Bio.SeqUtils as util
import collections as coll
import multiprocessing as mp
from functools import partial


HOST = "10.148.31.9"
PORT = 22
USER = "valentin-wesp"
PASSWORD = "OidaUokEidos?1871"
paramiko.util.log_to_file("paramiko_log.txt", level="INFO")


def create_ssh_client():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, PORT, USER, PASSWORD)
    sftp = ssh.open_sftp()
    return [ssh, sftp]


def get_aa_codon(id, prot_dna_dict, type, output, progress, size, lock):
    try:
        ssh,sftp = create_ssh_client()
        prot_path = prot_dna_dict[id][0]
        dna_path = prot_dna_dict[id][1]
        aa_codon_dct = coll.defaultdict(lambda: coll.defaultdict(int))
        gc_list = []
        with sftp.open(prot_path, "r") as prot_remote:
            with gzip.open(prot_remote, "rt") as prot_handle:
                prot_seqio = SeqIO.to_dict(SeqIO.parse(prot_handle, "fasta"))
                with sftp.open(dna_path, "r") as dna_remote:
                    with gzip.open(dna_remote, "rt") as dna_handle:
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
                                    if(len(dna_seq)/3 == prot_len-1):
                                        gc_list.append(util.gc_fraction(dna_seq))
                                        for i in range(prot_len):
                                            aa = prot_seq[i]
                                            codon = dna_seq[i*3:i*3+3]
                                            aa_codon_dct[codon][aa] += 1

        if(len(aa_codon_dct)):
            aa_codon_df = pd.DataFrame.from_dict(aa_codon_dct, orient="index")
            sorted_columns = sorted([col for col in aa_codon_df.columns])
            aa_codon_df = aa_codon_df[sorted_columns]
            aa_codon_df.index.name = "Amino acid"
            csv_buffer = io.StringIO()
            aa_codon_df.to_csv(csv_buffer, sep="\t")
            res_output = os.path.join(output, f"{id}_{np.mean(gc_list)}"+".csv")
            with sftp.open(res_output, "w") as csv_handle:
                csv_handle.write(csv_buffer.getvalue())

        sftp.close()
        ssh.close()
        with lock:
            progress.value += 1
            print("\r\033[2KProgress: {:.2f}% - last finished ID: {}".format(
                                        progress.value/size*100, id), end="")
    except paramiko.ssh_exception.SSHException:
        time.sleep(5)
        get_aa_codon(id, prot_dna_dict, type, output, progress, size, lock)


# main method
if __name__ == "__main__":
    print("Progress: {:.2f}% - last finished ID:".format(0), end="")

    path_to_data = sys.argv[1]
    output = sys.argv[2]
    type = sys.argv[3]
    processes = int(sys.argv[4])

    ssh,sftp = create_ssh_client()
    files = sftp.listdir(path_to_data)
    try:
        sftp.mkdir(output)
    except:
        pass

    sftp.close()
    ssh.close()

    mp.freeze_support()
    # set the multiprocessing manager
    manager = mp.Manager()
    # set the lock for multiple processes
    lock = manager.Lock()
    # set the multiprocessing variable to count the progress
    progress = manager.Value("i", 0)

    prot_ids = [os.path.basename(file).split(".fasta.gz")[0]
                for file in files
                if file.endswith(".fasta.gz")
                    and not file.endswith("_DNA.fasta.gz")]

    prot_dna_dict = manager.dict()
    for prot_id in prot_ids:
        prot_file = prot_id+".fasta.gz"
        dna_file = prot_id+"_DNA.fasta.gz"
        if(prot_file in files and dna_file in files):
            prot_dna_dict[prot_id] = [os.path.join(path_to_data, prot_file),
                                      os.path.join(path_to_data, dna_file)]

    id_list = list(prot_dna_dict.keys())
    # start the multicore process for a given number of cores
    with mp.Pool(processes=processes) as pool:
        # run the process for the given parameters
        pool_map = partial(get_aa_codon, prot_dna_dict=prot_dna_dict, type=type,
                           output=output, progress=progress, size=len(id_list),
                           lock=lock)
        process = pool.map_async(pool_map, id_list)
        pool.close()
        pool.join()
        process.get()

    print()
    os.remove("paramiko_log.txt")
