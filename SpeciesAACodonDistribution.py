import sys
import gzip
import pandas as pd
from Bio import SeqIO
import collections as col
import Bio.SeqUtils as util


def aa_codon_count(prot_seq, dna_seq):
    aa_to_codons = col.defaultdict(lambda: col.defaultdict(lambda: 0))
    prot_seq += "*"
    if(len(dna_seq) == len(prot_seq)*3):
        codons = [dna_seq[i:i+3] for i in range(0, len(dna_seq), 3)]
        for index in range(0, len(prot_seq), 1):
            aa_to_codons[prot_seq[index]][codons[index]] += 1

        return aa_to_codons

    return None


def get_aa_dis(prot_path, dna_path, output):
    all_aa_to_codons = col.defaultdict(lambda: col.defaultdict())
    with gzip.open(prot_path, "rt") as prot_handle:
        prot_seqio = SeqIO.to_dict(SeqIO.parse(prot_handle, "fasta"))
        prot_sp_seqio = {name.split("|")[1]:rec for name,rec in prot_seqio.items()
                         if name.startswith("sp")}
        with gzip.open(dna_path, "rt") as dna_handle:
            dna_seqio = SeqIO.to_dict(SeqIO.parse(dna_handle, "fasta"))
            dna_sp_seqio = {name.split("|")[1]:rec for name,rec in dna_seqio.items()
                            if name.startswith("sp")}

            for prot_id,prot_rec in prot_sp_seqio.items():
                dna_rec = dna_sp_seqio[prot_id]
                prot_seq = str(prot_rec.seq)
                dna_seq = str(dna_rec.seq)
                codon_counts = aa_codon_count(prot_seq, dna_seq)
                if(codon_counts != None):
                    for aa,codons in codon_counts.items():
                        codon_list = []
                        for codon,count in codons.items():
                            codon_list.append("{}:{}".format(codon, count))

                        all_aa_to_codons[prot_id][aa] = ";".join(codon_list)

                    all_aa_to_codons[prot_id]["Length"] = len(prot_rec.seq)
                    all_aa_to_codons[prot_id]["GC"] = util.gc_fraction(dna_seq)

    if(len(all_aa_to_codons)):
        aa_codon_df = pd.DataFrame.from_dict(all_aa_to_codons, orient="index")
        sorted_columns = sorted([col for col in aa_codon_df.columns
                                 if not col in ["GC", "Length"]])
        sorted_columns += ["GC", "Length"]
        aa_codon_df = aa_codon_df[sorted_columns]
        aa_codon_df.index.name = "Prot_ID"
        aa_codon_df.to_csv(output, sep="\t")


# main method
if __name__ == "__main__":
    path_to_prot = sys.argv[1]
    path_to_dna = sys.argv[2]
    output = sys.argv[3]

    get_aa_dis(path_to_prot, path_to_dna, output)
