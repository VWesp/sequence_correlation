import os
import tqdm
import argparse
import numpy as np
import pandas as pd
import scipy.stats as sci


# main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregating distribution files and their median statistics")
    parser.add_argument("-d", "--data", help="Specify the path to the folder with the distribution files", required=True)
    parser.add_argument("-o", "--output", help="Specify the output folder", required=True)
    args = parser.parse_args()
    
    data = args.data
    output= args.output
    
    frames = []
    num_prots = []
    for file in tqdm.tqdm(os.listdir(data), desc=f"Combining distribution files"):
    	tax_id = int(file.split(".csv")[0].split("_")[1])
    	df = pd.read_csv(os.path.join(data, file), sep="\t", header=0)
    	df.index = [tax_id] * len(df)
    	df.index.name = "TaxID"
    	frames.append(df)
    	num_prots.append(len(df))
    
    print("Aggregating data...")	
    aggregated_df = pd.concat(frames).fillna(0.0)
    
    print("Calculating medians...")	
    columns = aggregated_df.columns[3:]
    median_df = pd.DataFrame(index=["Median", "MAD"])
    median_df.loc["Median", "#Proteins"] = np.median(num_prots)
    median_df.loc["MAD", "#Proteins"] = sci.median_abs_deviation(num_prots)
    median_df.loc["Median", columns] = aggregated_df[columns].median()
    median_df.loc["MAD", columns] = (aggregated_df[columns] - median_df.loc["Median"]).abs().median()
    median_df.to_csv(os.path.join(output, "median_distributions.csv"), sep="\t")
