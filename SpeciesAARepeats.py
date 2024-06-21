import re
import sys
import gzip
import numpy as np
import pandas as pd
from Bio import SeqIO
import collections as col


def get_aa_reps(prot_path, min_len, distance, output):
    all_aa_reps = {}
    with gzip.open(prot_path, "rt") as prot_handle:
        prot_seqio = SeqIO.to_dict(SeqIO.parse(prot_handle, "fasta"))
        prot_sp_seqio = {name.split("|")[1]:rec for name,rec in prot_seqio.items()
                         if name.startswith("sp")}
        for prot_id,prot_rec in prot_sp_seqio.items():
            prot_seq = str(prot_rec.seq)
            aa_reps = seq_reps(prot_seq, min_len, distance)
            processed_aa_reps = col.defaultdict(lambda: 0)
            for aa,reps in aa_reps.items():
                for rep in reps:
                    processed_aa_reps[rep[2]] += 1

            processed_aa_reps["Length"] = len(prot_seq)
            all_aa_reps[prot_id] = processed_aa_reps

    if(len(all_aa_reps)):
        aa_rep_df = pd.DataFrame.from_dict(all_aa_reps, orient="index")
        sorted_columns = sorted(col for col in aa_rep_df.columns
                                if col != "Length")
        sorted_columns += ["Length"]
        aa_rep_df = aa_rep_df[sorted_columns]
        aa_rep_df.index.name = "Prot_ID"
        aa_rep_df.to_csv(output, sep="\t")



# method for getting all monorepeats in a sequence given a minimum length and
# the number of allowed non-matching characters between each hit
def seq_reps(seq, min_len, distance):
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


# main method
if __name__ == "__main__":
    path_to_prot = sys.argv[1]
    min_len = int(sys.argv[2])
    distance = int(sys.argv[3])
    output = sys.argv[4]

    get_aa_reps(path_to_prot, min_len, distance, output)
