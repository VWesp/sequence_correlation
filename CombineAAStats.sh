# Archaea tr
python3 CombineAAStats.py ../../mount/Archaea/abundances/tr/ corr_results/tr/Archaea/aa_corr_results.csv ../../mount/Archaea/encoding_data.csv genetic_codes/ code_map.csv 100000 60 &&
# Archaea sp
python3 CombineAAStats.py ../../mount/Archaea/abundances/sp/ corr_results/sp/Archaea/aa_corr_results.csv ../../mount/Archaea/encoding_data.csv genetic_codes/ code_map.csv 100000 60 &&

# Bacteria tr
python3 CombineAAStats.py ../../mount/Bacteria/abundances/tr/ corr_results/tr/Bacteria/aa_corr_results.csv ../../mount/Bacteria/encoding_data.csv genetic_codes/ code_map.csv 100000 60 &&
# Bacteria sp
python3 CombineAAStats.py ../../mount/Bacteria/abundances/sp/ corr_results/sp/Bacteria/aa_corr_results.csv ../../mount/Bacteria/encoding_data.csv genetic_codes/ code_map.csv 100000 60 &&

# Eukaryotes tr
python3 CombineAAStats.py ../../mount/Eukaryotes/abundances/tr/ corr_results/tr/Eukaryotes/aa_corr_results.csv ../../mount/Eukaryotes/encoding_data.csv genetic_codes/ code_map.csv 100000 60 &&
# Eukaryotes sp
python3 CombineAAStats.py ../../mount/Eukaryotes/abundances/sp/ corr_results/sp/Eukaryotes/aa_corr_results.csv ../../mount/Eukaryotes/encoding_data.csv genetic_codes/ code_map.csv 100000 60 &&

# Viruses tr
python3 CombineAAStats.py ../../mount/Viruses/abundances/tr/ corr_results/tr/Viruses/aa_corr_results.csv ../../mount/Viruses/encoding_data.csv genetic_codes/ code_map.csv 100000 60 &&
# Viruses sp
python3 CombineAAStats.py ../../mount/Viruses/abundances/sp/ corr_results/sp/Viruses/aa_corr_results.csv ../../mount/Viruses/encoding_data.csv genetic_codes/ code_map.csv 100000 60
