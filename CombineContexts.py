import os
import sys
import numpy as np
import pandas as pd


if __name__ == "__main__":
    data,output,type = sys.argv[1:4]

    for step in range(1, 10, 1):
        step_output = os.path.join(output, str(step))
        step_data = os.path.join(data, str(step))
        paths = [os.path.join(step_data, file)
                 for file in os.listdir(step_data)]

        dataframes = [pd.read_csv(path, sep="\t", header=0, index_col=0)
                      for path in paths]
        rows = dataframes[0].index
        combined_df = pd.concat(dataframes)

        type_output = os.path.join(step_output, type)
        os.makedirs(type_output, exist_ok=True)

        comb_med_df = combined_df.groupby(combined_df.index).agg("median")
        comb_med_df = comb_med_df.reindex(index=rows)
        res_output = os.path.join(type_output, f"{type}_context_median.csv")
        comb_med_df.to_csv(res_output, sep="\t")

        comb_sum_df = combined_df.groupby(combined_df.index).agg("sum")
        comb_sum_df = comb_sum_df.reindex(index=rows)
        res_output = os.path.join(type_output, f"{type}_context_sum.csv")
        comb_sum_df.to_csv(res_output, sep="\t")
