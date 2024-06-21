import os
import sys
import pandas as pd


if __name__ == "__main__":
    path_to_data = sys.argv[1]
    output = sys.argv[2]

    data_files = [os.path.join(path_to_data, file)
                  for file in os.listdir(path_to_data)
                  if file.endswith(".csv")]

    dataframes = []
    for file in data_files:
        id = os.path.basename(file).split(".csv")[0]
        dataframe = pd.read_csv(file, sep="\t", index_col=0)
        dataframe["Genome_Tax_ID"] = id
        dataframes.append(dataframe)

    combined_frame = pd.concat(dataframes)
    combined_frame = combined_frame.fillna(0)
    sorted_columns = sorted([col for col in combined_frame.columns
                             if not col in ["Length", "Genome_Tax_ID"]])
    sorted_columns += ["Length", "Genome_Tax_ID"]
    combined_frame = combined_frame[sorted_columns]
    combined_frame.to_csv(output, sep="\t")
