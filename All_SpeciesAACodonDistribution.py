import os
import sys
import multiprocessing as mp
from functools import partial
import SpeciesAACodonDistribution as saacd


def main(id, prot_dna_dict, output, progress, size, lock):
    prot_path = prot_dna_dict[id][0]
    dna_path = prot_dna_dict[id][1]
    res_output = os.path.join(output, id+".csv")
    saacd.get_aa_dis(prot_path, dna_path, res_output)
    with lock:
        progress.value += 1

    print("\rProgress: {:.2f}% - last finished ID: {}{}".format(
                                   progress.value/size*100, id, " "*20), end="")

# main method
if __name__ == "__main__":
    path_to_data = sys.argv[1]
    output = sys.argv[2]
    processes = int(sys.argv[3])

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

    # start the multicore process for a given number of cores
    with mp.Pool(processes=processes) as pool:
        # run the process for the given parameters
        pool_map = partial(main, prot_dna_dict=prot_dna_dict, output=output,
                           progress=progress, size=len(prot_dna_dict.keys()),
                           lock=lock)
        process = pool.map_async(pool_map, list(prot_dna_dict.keys()))
        pool.close()
        #print(process.get())
        pool.join()

    print()
