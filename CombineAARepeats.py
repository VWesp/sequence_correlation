import os
import sys
import json
import numpy as np
import pandas as pd
import collections as col


if __name__ == "__main__":
    path,output = sys.argv[1:3]

    os.makedirs(output, exist_ok=True)

    file_paths = [os.path.join(path, file) for file in os.listdir(path)]
    agg_data = col.defaultdict(lambda: col.defaultdict(lambda: col.defaultdict(lambda: col.defaultdict(int))))
    for file in file_paths:
        prot_data = col.defaultdict(lambda: col.defaultdict(int))
        prot_id = os.path.basename(file).split(".")[0]
        with open(file, "r") as reader:
            json_data = json.load(reader)
            for id,aa_data in json_data.items():
                for aa,rep_data in aa_data.items():
                    for rep_len,num in rep_data.items():
                        prot_data[aa][int(rep_len)] += num

        for aa,rep_data in prot_data.items():
            for rep_len,num in rep_data.items():
                agg_data[prot_id][aa][rep_len] = num

    with open(os.path.join(output, "proteome_data.json"), "w", encoding="utf-8") as writer:
        json.dump(agg_data, writer, ensure_ascii=False, indent=4)

    sum_data = col.defaultdict(lambda: col.defaultdict(int))
    for _,aa_data in agg_data.items():
        for aa,rep_data in aa_data.items():
            for rep_len,num in rep_data.items():
                sum_data[aa][rep_len] += num

    sum_df = pd.DataFrame.from_dict(sum_data).fillna(0).sort_index()
    sum_df.to_csv(os.path.join(output, "proteome_data.csv"), sep="\t")
