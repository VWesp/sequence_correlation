import os
import sys
import numpy as np
import pandas as pd


if __name__ == "__main__":
    data,output = sys.argv[1:3]

    for step in range(1, 10, 1):
        step_output = os.path.join(output, str(step))
        os.makedirs(step_output, exist_ok=True)
        step_data = os.path.join(data, str(step))
        paths = [os.path.join(step_data, file) for file in os.listdir(step_data)]
        dataframes = [pd.read_csv(df, sep="\t", header=0, index_col=0)
                      for df in paths]
        combined_df = pd.concat(dataframes)
        comb_med_df = combined_df.groupby(combined_df.index).agg("median")

        res_output = os.path.join(step_output, "codon_context.csv")
        comb_med_df.to_csv(res_output, sep="\t")
