import re
import os
import sys
import gzip


# main method
if __name__ == "__main__":
    path_to_file = sys.argv[1]
    output = sys.argv[2]

    tax_ids = set()
    re_str = "NCBI_TaxID=(.*);"
    with gzip.open(path_to_file, "rt", encoding="utf-8") as handle:
        for line in handle.readlines():
            tax_id = re.search(re_str, line.strip())
            if(tax_id != None):
                match = re.split("=", tax_id.group())[1]
                match = re.split("(;| )", match)[0]
                tax_ids.add(match)

    with open(output, "w") as handle:
        handle.write("Tax_ID\n"+"\n".join(tax_ids))
