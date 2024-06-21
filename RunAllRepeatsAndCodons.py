import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
import RepeatsAndCodons as rac


# method for running the multiprocessing given the path to the species folder,
# the number of species, the minimum distance between each repeat, the minimum
# repeat length, the progress value and the lock
def main(file, size, min_len, distance, output, type_ids, progress, lock):
    # the name of the species
    prot_id = os.path.basename(file).split(".")[0]
    if(type_ids != None):
        tax_id = int(prot_id.split("_")[1])
        found_id = False
        for type,ids in type_ids.items():
            if(tax_id in ids):
                output = os.path.join(output, type)
                found_id = True
                break

        if(not found_id):
            return

    prot_records = rac.readSeqFile(file)
    # dictionary containing the data for the repeats
    repeat_data = {}
    # dictionary containing the data for the amino acids and their numbers
    aa_num_data = {}
    # loop over all proteins and their records
    for name,rec in prot_records[0].items():
        id = name.split("|")[1]
        # set all characters in the protein sequence to upper
        prot_seq = str(rec.seq).upper()

        # get the repeat data for each protein
        repeat_data[id] = rac.getMonoRepeats(prot_seq, min_len, distance)

        # get the amino acid/codon data for each protein
        aa_num_data[id] = rac.getProteinCodonDistribution(prot_seq)

    # path to the output of the repeat data
    rep_output = os.path.join(os.path.join(output, "monorepeats"),
                              prot_id+"_monorepeats.csv")
    # save the repeat data
    rac.saveRepeats(repeat_data, rep_output)

    # path to the output of the amino acid/codon data
    aa_num_output = os.path.join(os.path.join(output, "aa_distribution"),
                                 prot_id+"_aa_distribution.csv")
    # save the amino acid/codon data
    rac.saveDistributions(aa_num_data, aa_num_output)

    prot_records[1].close()

    with lock:
        progress.value += 1

    print("\rProgress: {:.2f}% - last finished proteome: {}{}".format(
                              progress.value/size*100, prot_id, " "*50), end="")


# main method
if __name__ == "__main__":
    # path to the folder with the proteomes
    path_to_proteomes = sys.argv[1]
    # path to output folder
    output = sys.argv[2]
    # the minimum repeat length
    rep_min_len = int(sys.argv[3])
    # the minimum distance between each repeat
    rep_distance = int(sys.argv[4])
    # the number of processes to be used
    processes = int(sys.argv[5])
    #
    tax_ids = int(sys.argv[6])

    os.makedirs(output, exist_ok=True)

    type_ids = None
    if(tax_ids):
        type_ids = {}
        types = ["fungi", "human", "invertebrates", "mammals", "plants",
                 "rodents", "vertebrates"]
        for type in types:
            tax_df = pd.read_csv(type+"_ids.csv")
            type_ids[type] = np.asarray(pd.to_numeric(tax_df["Tax_ID"]))
            os.makedirs(os.path.join(os.path.join(output, type), "monorepeats"),
                        exist_ok=True)
            os.makedirs(os.path.join(os.path.join(output, type),
                        "aa_distribution"), exist_ok=True)
    else:
        os.makedirs(os.path.join(output, "monorepeats"), exist_ok=True)
        os.makedirs(os.path.join(output, "aa_distribution"), exist_ok=True)


    mp.freeze_support()
    # set the multiprocessing manager
    manager = mp.Manager()
    # set the lock for multiple processes
    lock = manager.Lock()
    # set the multiprocessing variable to count the progress
    progress = manager.Value("d", 0)

    proteomes = [os.path.join(path_to_proteomes, file)
                 for file in os.listdir(path_to_proteomes)
                 if file.endswith(".fasta.gz")]
    # start the multicore process for a given number of cores
    with mp.Pool(processes=processes) as pool:
        # run the process for the given parameters
        pool_map = partial(main, size=len(proteomes), min_len=rep_min_len,
                           distance=rep_distance, output=output,
                           type_ids=type_ids, progress=progress, lock=lock)
        process = pool.map_async(pool_map, proteomes)
        pool.close()
        #print(process.get())
        pool.join()

    print()
