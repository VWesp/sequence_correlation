import os
import sys
import numpy as np
import pandas as pd
import collections as col


if __name__ == "__main__":
    # path to the folder with the CSV files
    path_to_folder = sys.argv[1]
    # path to the output file
    output = sys.argv[2]

    os.makedirs(output, exist_ok=True)

    csv_files = [os.path.join(path_to_folder, file)
                 for file in os.listdir(path_to_folder)
                 if file.endswith("_monorepeats.csv")]

    progress = 0
    prog_len = len(csv_files)
    print("\rPlot progress: {:.2f}%".format(progress/prog_len*100), end="")
    mono_reps = col.defaultdict(lambda: col.defaultdict(lambda: [[], []]))
    lengths = set()
    for file in csv_files:
        df = pd.read_csv(file, sep="\t")
        for index,row in df.iterrows():
            rep_amino_acid = row["Amino acid:Length"].split(":")[0]
            rep_length = int(row["Amino acid:Length"].split(":")[1])
            lengths.add(rep_length)
            rep_num = int(row["#Repeats"])
            proteins = row["Proteins"]
            mono_reps[rep_amino_acid][rep_length][0].append(rep_num)
            mono_reps[rep_amino_acid][rep_length][1].append(proteins)

        progress += 1
        print("\rPlot progress: {:.2f}%".format(progress/prog_len*100), end="")

    print()
    sum_mono_reps = col.defaultdict(lambda: col.defaultdict())
    mean_mono_reps = col.defaultdict(lambda: col.defaultdict())
    for aa,dic in mono_reps.items():
        for length,reps in dic.items():
            sum_mono_reps[aa][length] = np.sum(reps[0])
            mean_mono_reps[aa][length] = np.mean(reps[0])

    sum_out_df = pd.DataFrame.from_dict(sum_mono_reps, orient="index")
    sum_out_df = sum_out_df[list(lengths)]
    sum_out_df = sum_out_df.fillna(0)
    sum_out_df.to_csv(os.path.join(output, "sum_monorepeats.csv"), sep="\t")

    mean_out_df = pd.DataFrame.from_dict(mean_mono_reps, orient="index")
    mean_out_df = mean_out_df[list(lengths)]
    mean_out_df = mean_out_df.fillna(0)
    mean_out_df.to_csv(os.path.join(output, "mean_monorepeats.csv"), sep="\t")
