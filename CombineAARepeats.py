import os
import sys
import json
import numpy as np
import collections as col


if __name__ == "__main__":
    path,output = sys.argv[1:3]

    os.makedirs(output, exist_ok=True)

    file_paths = [os.path.join(path, file) for file in os.listdir(path)]
    agg_data = col.defaultdict(lambda: col.defaultdict(lambda: col.defaultdict(lambda: col.defaultdict(float))))
    for file in file_paths:
        prot_data = col.defaultdict(lambda: col.defaultdict(list))
        prot_id = os.path.basename(file).split(".")[0]
        with open(file, "r") as reader:
            json_data = json.load(reader)
            for id,aa_data in json_data.items():
                for aa,rep_data in aa_data.items():
                    for rep_len,num in rep_data.items():
                        prot_data[aa][int(rep_len)].append(num)

        for aa,rep_data in prot_data.items():
            for rep_len,num_data in rep_data.items():
                agg_data[prot_id][aa][rep_len]["sum"] = float(np.sum(num_data))
                agg_data[prot_id][aa][rep_len]["mean"] = float(np.mean(num_data))
                agg_data[prot_id][aa][rep_len]["std"] = float(np.std(num_data))

    with open(os.path.join(output, "proteome_data.json"), "w", encoding="utf-8") as writer:
        json.dump(agg_data, writer, ensure_ascii=False, indent=4)
