import os
import sys
import yaml


if __name__=="__main__":
    path,output = sys.argv[1:3]
    os.makedirs(output, exist_ok=True)

    letter_code = {"A": 1, "C": 2, "G": 4, "T": 8}
    iupac = {1: "A", 2: "C", 3: "M", 4: "G", 5: "R", 6: "S", 7: "V", 8: "T",
             9: "W", 10: "Y", 11: "H", 12: "K", 13: "D", 14: "B", 15: "N"}
    for code in os.listdir(path):
        if(code.endswith(".yaml") and not code.endswith("deg.yaml")):
            yaml_code = None
            with open(os.path.join(path, code), "r") as code_reader:
                yaml_code = yaml.safe_load(code_reader)

            deg_code = {}
            for aa,codons in yaml_code.items():
                deg_code[aa] = []
                two_list = set()
                for codon in yaml_code[aa]:
                    if(isinstance(codon, dict)):
                        deg_code[aa].append(codon)
                    else:
                        two_list.add(codon[:2])

                for two_cod in two_list:
                    encoding = 0
                    for codon in yaml_code[aa]:
                        if(not isinstance(codon, dict) and codon.startswith(two_cod)):
                            encoding += letter_code[codon[2]]

                    deg_code[aa].append(f"{two_cod}{iupac[encoding]}")

            with open(os.path.join(output, code), "w") as code_writer:
                yaml.dump(deg_code, code_writer, default_flow_style=False)
