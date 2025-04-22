import sympy as sp
import numpy as np
import math
import re

# method for building the frequency function of each codon and each amino acid
# given a genetic code in yaml format
def build_functions(code):
    # frequencies of each nucleotide depending on the GC content
    g = sp.symbols("g", float=True)
    letter_func = {}
    letter_func["A"] = (1-g)/2
    letter_func["C"] = g/2
    letter_func["G"] = g/2
    letter_func["T"] = (1-g)/2

    # additional nucleotide symbols (IUPAC)
    letter_func["R"] = letter_func["A"] + letter_func["G"]
    letter_func["Y"] = letter_func["C"] + letter_func["T"]
    letter_func["S"] = letter_func["C"] + letter_func["G"]
    letter_func["W"] = letter_func["A"] + letter_func["T"]
    letter_func["K"] = letter_func["G"] + letter_func["T"]
    letter_func["M"] = letter_func["A"] + letter_func["C"]
    letter_func["B"] = 1 - letter_func["A"]
    letter_func["D"] = 1 - letter_func["C"]
    letter_func["H"] = 1 - letter_func["G"]
    letter_func["V"] = 1 - letter_func["T"]
    letter_func["N"] = 1

    funcs = {"codon": {}, "amino": {}}
    #loop over all amino acids and their codons

    for aminoacid,codons in code.items():
        amino_func = 0
        # loop over each codon
        for codon in codons:
            codon_func = 1
            # if the codon appears for several amino acids (or stop), the
            # frequencies for this codon are adjusted according to the yaml file
            adjusted_freq = 1
            if(type(codon) is dict):
                codon_temp = next(iter(codon))
                try:
                    adjusted_freq = float(codon[codon_temp])
                except ValueError:
                    nom, denom = codon[codon_temp].split("/")
                    adjusted_freq = float(nom) / float(denom)

                codon = codon_temp

            # build the codon frequency given the GC formula
            for letter in codon:
                codon_func *= letter_func[letter]

            # adjust the codon frequency
            codon_func *= adjusted_freq
            # save the codon frequency function
            funcs["codon"][codon] = codon_func
            # build the amino acid frequency given each codon function
            amino_func += codon_func

        # save the amino frequency function
        funcs["amino"][aminoacid] = amino_func

    return funcs

# method for calculating the frequency of each codon/amino acid given their
# frequency functions and a GC content
def calculate_frequencies(funcs, g_content=0.5):
    g = sp.symbols("g", float=True)
    freqs = {"codon": {}, "amino": {}}
    # loop over all  codon frequency functions and calculate their frequency for
    # a given GC content
    for codon,func in funcs["codon"].items():
        freqs["codon"][codon] = float(func.subs(g, g_content))

    # loop over all amino frequency functions and calculate their frequency for
    # a given GC content
    for codon,func in funcs["amino"].items():
        freqs["amino"][codon] = float(func.subs(g, g_content))

    return freqs

# method for generating random codon sequences given codon/amino acid frequencies
def generate_random_sequences(freqs, funcs, length=100, number=1, seed=None):
    rng = np.random.default_rng(seed)
    # get all codons
    codons = list(freqs["codon"].keys())
    # get all their frequencies
    codon_freqs = np.asarray(list(freqs["codon"].values()))
    codon_freqs /= codon_freqs.sum()
    # randomly draw codons (length) multiple times (number)
    codon_rand_seqs = rng.choice(a=codons, size=length*number,
                                 p=codon_freqs).reshape(number, length)
    # join codon sequences together
    codon_rand_seqs = ["".join(seq) for seq in codon_rand_seqs]

    # get all amino acids
    aminos = list(freqs["amino"].keys())
    # get all their frequencies
    amino_freqs = np.asarray(list(freqs["amino"].values()))
    amino_freqs /= amino_freqs.sum()
    # randomly draw amino acids (length) multiple times (number)
    amino_rand_seqs = rng.choice(a=aminos, size=length*number,
                                 p=amino_freqs).reshape(number, length)
    # join amino acid sequences together
    amino_rand_seqs = ["".join(seq) for seq in amino_rand_seqs]

    return [codon_rand_seqs, amino_rand_seqs]

# method for creating the generalized topological entropy table for the equation
# 4^n - n + 1 <= |s| < 4^(n+1) - (n + 1) + 1
# additionally, calculate the expected gte for each n
# the '4' in the gte table calculation indicates an alphabet of size 4
def get_gte_table(a_size=4):
    # calculate the max_len so that 4^n - n + 1 <= |s|
    gte_table = [0] + [a_size**n + n - 1 for n in range(1, 20)]
    return np.asarray(gte_table)

# method for calculating the generalized topological entropy of a sequence
# the '4' in the gte table calculation indicates an alphabet of size 4
def calculate_gte(seq, gte_table, a_size=4):
    # we take the highest n and subsequence length for which the equation
    # 4^n - n + 1 <= |s| holds true
    largest_n = np.where(gte_table<=len(seq))[0][-1]
    sub_seq_len = gte_table[largest_n]

    # generalized topological entropy equation h = log4(p_w(n)) / n
    # for all subsequences of length 4^n - n + 1 <= |s| (no overlaps for faster
    # run time)
    # p_w(n) are the number of different subwords of length n in the subsequence
    # (with overlaps)
    # afterwards take the median of all subsequences
    gte_entropies = []
    # if the sequence is too short for this alphabet, no entropy can be calculated
    if(largest_n > 0):
        for sub_seq in re.finditer("."*sub_seq_len+"?", seq):
            sub_seq = sub_seq.group()
            sub_words = set([sub_seq[i:i+largest_n] for i in
                             range(len(sub_seq)-largest_n+1)])
            gte_entropies.append(math.log(len(sub_words), a_size) / largest_n)
    else:
        return 0

    return np.median(np.asarray(gte_entropies))
