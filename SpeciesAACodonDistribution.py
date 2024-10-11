import sys
import gzip
import pandas as pd
from Bio import SeqIO
import collections as col
import Bio.SeqUtils as util


def get_aa_dis(prot_path, dna_path, output):
    all_aa_count = col.defaultdict(lambda: col.defaultdict())
    with gzip.open(prot_path, "rt") as prot_handle:
        prot_seqio = SeqIO.to_dict(SeqIO.parse(prot_handle, "fasta"))
        with gzip.open(dna_path, "rt") as dna_handle:
            dna_seqio = SeqIO.to_dict(SeqIO.parse(dna_handle, "fasta"))
            dna_seqio = {id.split("|")[1]:rec for id,rec in dna_seqio.items()}
            for prot_id,prot_rec in prot_seqio.items():
                id = prot_id.split("|")[1]
                if(id in dna_seqio):
                    dna_rec = dna_seqio[id]
                    prot_seq = str(prot_rec.seq)
                    prot_len = len(prot_rec.seq)
                    dna_seq = str(dna_rec.seq)
                    amino_acids = set(prot_seq)
                    all_aa_count[id] = {aa:prot_seq.count(aa)/prot_len
                                        for aa in amino_acids}
                    all_aa_count[id]["Length"] = prot_len
                    all_aa_count[id]["GC"] = util.gc_fraction(dna_seq)
                    all_aa_count[id]["Status"] = prot_id.split("|")[0]

    if(len(all_aa_count)):
        aa_codon_df = pd.DataFrame.from_dict(all_aa_count, orient="index")
        sorted_columns = sorted([col for col in aa_codon_df.columns
                                 if not col in ["GC", "Length", "Status"]])
        aa_codon_df = aa_codon_df[sorted_columns+["GC", "Length", "Status"]]
        aa_codon_df.index.name = "Prot_ID"
        aa_codon_df.to_csv(output, sep="\t")


# main method
if __name__ == "__main__":
    path_to_prot = sys.argv[1]
    path_to_dna = sys.argv[2]
    output = sys.argv[3]

    get_aa_dis(path_to_prot, path_to_dna, output)
