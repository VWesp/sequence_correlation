import os
import sys
import numpy as np
import pandas as pd


files = [os.path.join(sys.argv[1], file)
         for file in os.listdir(sys.argv[1])]

print(f"Organisms: {len(files)}")
num_prots = sum([len(pd.read_csv(file, sep="\t", header=0, index_col=0))
                 for file in files])
print(f"Proteins: {num_prots}")
