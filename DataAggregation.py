import os
import tqdm
import argparse
import pandas as pd


# main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregating distribution files and their median statistics")
    parser.add_argument("-d", "--data", help="Specify the path to the folder with the distribution files", required=True)
    parser.add_argument("-o", "--output", help="Specify the output folder", required=True)
    args = parser.parse_args()
    
    data = args.data
    output= args.output
    
    frames = []
    for file in tqdm.tqdm(os.listdir(data), desc=f"Combining distribution files"):
    	tax_id = int(file.split(".csv")[0].split("_")[1])
    	df = pd.read_csv(os.path.join(data, file), sep="\t", header=0)
    	df.index = [tax_id] * len(df)
    	df.index.name = "TaxID"
    	frames.append(df)
    	
    aggregated_df = pd.concat(frames)
    aggregated_df.to_csv(os.path.join(output, "aggregated_distributions.csv"), sep="\t")
    
    median_df = pd.DataFrame(columns=aggregated_df.columns[2:], index=["Median", "MAD"])
    median_df.loc["Median"] = aggregated_df.median()
    median_df.loc["MAD"] = (aggregated_df - median_df.loc["Median"]).abs().median()
    median_df.to_csv(os.path.join(output, "median_distributions.csv"), sep="\t")
