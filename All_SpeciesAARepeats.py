import os
import sys
import multiprocessing as mp
from functools import partial
import SpeciesAARepeats as saar


def main(prot_path, min_len, distance, output, progress, size, lock):
    id = os.path.basename(prot_path).split(".fasta.gz")[0]
    res_output = os.path.join(output, id+".csv")
    saar.get_aa_reps(prot_path, min_len, distance, res_output)
    with lock:
        progress.value += 1

    print("\rProgress: {:.2f}% - last finished ID: {}{}".format(
                                   progress.value/size*100, id, " "*20), end="")

# main method
if __name__ == "__main__":
    path_to_data = sys.argv[1]
    min_len = int(sys.argv[2])
    distance = int(sys.argv[3])
    output = sys.argv[4]
    processes = int(sys.argv[5])

    os.makedirs(output, exist_ok=True)

    mp.freeze_support()
    # set the multiprocessing manager
    manager = mp.Manager()
    # set the lock for multiple processes
    lock = manager.Lock()
    # set the multiprocessing variable to count the progress
    progress = manager.Value("d", 0)

    prot_files = [os.path.join(path_to_data, file)
                  for file in os.listdir(path_to_data)
                  if file.endswith(".fasta.gz")
                    and not file.endswith("_DNA.fasta.gz")]

    # start the multicore process for a given number of cores
    with mp.Pool(processes=processes) as pool:
        # run the process for the given parameters
        pool_map = partial(main, min_len=min_len, distance=distance,
                           output=output, progress=progress,
                           size=len(prot_files), lock=lock)
        process = pool.map_async(pool_map, prot_files)
        pool.close()
        #print(process.get())
        pool.join()

    print()
