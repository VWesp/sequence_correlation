import os
import tqdm
import argparse
import numpy as np
import pandas as pd
import scipy.stats as sci


# main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the overall median of a domain")
    parser.add_argument("-d", "--data", help="Specify the path to the folder with the distribution files", required=True)
    parser.add_argument("-o", "--output", help="Specify the output file", required=True)
    args = parser.parse_args()
    
    data = args.data
    output= args.output
    
    columns = ["Length", "GC", "M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I", "A", "G", "P", "T", "V", "L", "R", "S"]
    combined_df = pd.DataFrame(columns=columns, index=["Median", "MAD"])
    for col in columns:
        values = []
        for file in tqdm.tqdm(os.listdir(data), desc=f"Combining distribution values for column {col}"):
            df = pd.read_csv(os.path.join(data, file), sep="\t", header=0, index_col=0)
            try:
                values.extend(df[col].astype(float).fillna(0.0).values)
            except KeyError as ke:
                values.extend([0.0]*len(df))
            
        combined_df.loc["Median", col] = np.median(values)
        combined_df.loc["MAD", col] = sci.median_abs_deviation(values)
            
    combined_df.to_csv(output, sep="\t")
