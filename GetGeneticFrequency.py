import os
import sys
import yaml
import pandas as pd


# One letter code for the amino acids of the genetic codes without Stop
AA_TO_ONE_LETTER = {"Methionine": "M", "Threonine": "T", "Asparagine": "N",
                    "Lysine": "K", "Serine": "S", "Arginine": "R",
                    "Valine": "V", "Alanine": "A", "Aspartic_acid": "D",
                    "Glutamic_acid": "E", "Glycine": "G", "Phenylalanine": "F",
                    "Leucine": "L", "Tyrosine": "Y", "Cysteine": "C",
                    "Tryptophane": "W", "Proline": "P", "Histidine": "H",
                    "Glutamine": "Q", "Isoleucine": "I"}


# main method
if __name__ == "__main__":
    code_dir,output = sys.argv[1:3]
    os.makedirs(output, exist_ok=True)

    codes = [file for file in os.listdir(code_dir)
             if file.endswith(".yaml")]

    for code in codes:
        code_path = os.path.join(code_dir, code)
        with open(code_path, "r") as code_reader:
            yaml_code = yaml.safe_load(code_reader)
            freq_map = {}
            for aa,one_letter in AA_TO_ONE_LETTER.items():
                freq_map[one_letter] = 0.0
                codon_map = yaml_code[aa]
                for codon in codon_map:
                    if(isinstance(codon, str)):
                        freq_map[one_letter] += 1
                    else:
                        nom,denom = list(codon.values())[0].split("/")
                        freq_map[one_letter] += float(nom) / float(denom)

            total_codon_num = sum([num for num in freq_map.values()])
            for aa,codon_num in freq_map.items():
                freq_map[aa] /= total_codon_num

            freq_df = pd.DataFrame.from_dict(freq_map, orient="index")
            freq_df.index.name = "AminoAcid"
            freq_df.columns = ["Frequency"]
            freq_output = os.path.join(code_dir, f"{code.split(".")[0]}.csv")
            freq_df.to_csv(freq_output, sep="\t")
