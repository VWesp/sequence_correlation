import re
import os
import sys
import gzip
import numpy as np
import pandas as pd
from Bio import SeqIO
import collections as col

'''# the frequency of each amino acid dependent on the GC content
g = sp.symbols("g", float=True)
aa_freq = {"N": (1-g)**2/8, "K": (1-g)**2/8, "F": (1-g)**2/8, "Y": (1-g)**2/8,
           "L": (1-g**2)/8, "I": ((1-g)**2*(2-g))/8, "M": (g*(1-g)**2)/8,
           "D": (g*(1-g))/8, "C": (g*(1-g))/8, "Q": (g*(1-g))/8,
           "E": (g*(1-g))/8, "H": (g*(1-g))/8, "T": (g*(1-g))/4,
           "V": (g*(1-g))/4, "S": (3*g*(1-g))/8, "W": (g**2*(1-g))/8,
           "A": g**2/4, "G": g**2/4, "P": g**2/4, "R": (g*(1+g))/8}'''

# method for reading a zipped sequence file and returning it as a dictionary
def readSeqFile(path):
    # open the zipped proteome file
    handle = gzip.open(path, "rt")
    # get the records of each protein
    return [SeqIO.to_dict(SeqIO.parse(handle, "fasta")), handle]


# method for getting all monorepeats in a sequence given a minimum length and
# the number of allowed non-matching characters between each hit
def getMonoRepeats(seq, min_len, distance):
    # dictionary containing the positions and sequence of each repeat
    merged_reps = col.defaultdict(lambda: [])
    # loop over all characters in a sequence
    for element in set(seq):
        # get all subsequences containing only the current character
        matches = re.finditer(r"{}+".format(element), seq, flags=re.IGNORECASE)
        # filter each subsequence based on their length and save their start
        # and end positions
        repeats = np.array([[match.start(), match.end()] for match in matches
                            if len(match.group()) >= min_len])
        # only merge the repeats if we have at least one repeat
        if(len(repeats)):
            # start position of the first repeat (global)
            rep_start = repeats[0][0]
            # end position of the first repeat (global)
            rep_end = repeats[0][1]
            # loop over all subsequent repeats and merge them if positions
            # overlap
            for reps in repeats[1:]:
                # if the number of non-matching characters between the start
                # of the current repeat and the end of the last repeat (global)
                # is not greater than the given allowed distance, set the end of
                # the last repeats (global) to the end of the current repeat
                # else, save the positions of the merged repeats and start a
                # new repeat by setting the global start and end positions to
                # the current start and end positions
                if(reps[0]-rep_end <= distance):
                    rep_end = reps[1]
                else:
                    merged_reps[element].append([rep_start, rep_end,
                                                 seq[rep_start:rep_end]])
                    rep_start = reps[0]
                    rep_end = reps[1]

            # save the start and end positions of the last repeat in the
            # sequence
            merged_reps[element].append([rep_start, rep_end,
                                         seq[rep_start:rep_end]])

    # return all merged repeats
    return merged_reps


# method for returning the codons and their numbers for each amino acid in a
# protein sequence given the CDS sequence
def getProteinCodonDistribution(prot_seq):
    # add the stop symbol to the end of the protein sequence
    prot_seq += "*"
    # dictionary containing the type and number of amino acids in a protein
    # sequence
    aa_number = {aa:prot_seq.count(aa) for aa in set(prot_seq)}
    # return all amino acids numbers for the protein sequence
    return aa_number


# method for saving the results of the repeat function
def saveRepeats(data, output):
    # dictionary for containing the combined repeat data
    combined_data = col.defaultdict(lambda: [set(), 0, 0, set()])
    # loop over all proteins and their repeats
    for id,repeats in data.items():
        # loop over all amino acids and their repeats
        for aa,info in repeats.items():
            # loop over all repeats
            for rep in info:
                rep_seq = rep[2]
                amino_acid = rep_seq[0]
                amount_aa = rep_seq.count(amino_acid)
                rep_key = "{}:{}".format(amino_acid, amount_aa)
                combined_data[rep_key][0].add(rep_seq)
                # add the found number of the chosen repeat
                combined_data[rep_key][1] += 1
                # add the protein IDs containing the chosen repeat
                combined_data[rep_key][3].add(id)

    # loop again over all repeats and add additional information
    for rep in combined_data:
        combined_data[rep][0] = ";".join(combined_data[rep][0])
        # add the number of proteins containing the chosen repeat
        combined_data[rep][2] = len(combined_data[rep][3])
        # join the protein IDs containing the chosen repeat to a single string
        combined_data[rep][3] = ";".join(combined_data[rep][3])

    # sort the data after the repeats and save it to a CSV file
    sorted_data = dict(sorted(combined_data.items()))
    dataframe = pd.DataFrame.from_dict(sorted_data, orient="index").reset_index()
    dataframe.to_csv(output, sep="\t", index=False,
              header=["Amino acid:Length", "Repeat", "#Repeats",
                                "#Proteins ({})".format(len(data)), "Proteins"])


# method for saving the results of the amino acids/codon function
def saveDistributions(data, output):
    # save the data to a CSV file
    dataframe = pd.DataFrame.from_dict(data, orient="index").reset_index()
    dataframe.rename(columns={dataframe.columns[0]: "Proteins"}, inplace=True)
    dataframe = dataframe.fillna(0)
    dataframe["Sum"] = dataframe.iloc[:, 1:].apply(np.sum, axis=1)
    dataframe.to_csv(output, sep="\t", index=False)


# main method
if __name__ == "__main__":
    # read the zipped protein file
    prot_records = readSeqFile(sys.argv[1])
    # dictionary containing the data for the repeats
    repeat_data = {}
    # dictionary containing the data for the amino acids and their numbers
    dis_data = {}
    # loop over all proteins and their records
    for name,rec in prot_records[0].items():
        id = name.split("|")[1]
        # get the repeat data for each protein
        repeat_data[id] = getMonoRepeats(str(rec.seq), int(sys.argv[2]),
                                                           int(sys.argv[3]))
        # get the amino acid/codon data for each protein
        dis_data[id] = getProteinCodonDistribution(str(rec.seq))

    # close the zipped protein file
    prot_records[1].close()
    # save the repeat data
    saveRepeats(repeat_data, sys.argv[4])
    # save the amino acid/codon data
    saveDistributions(dis_data, sys.argv[5])
