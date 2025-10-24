#!/bin/bash

for domain in "Archaea" "Bacteria" "Eukaryota" "Viruses"; do
	echo $domain
	#/root/anaconda3/bin/python3 DownloadProteome.py -d $domain -o /home/vwesp/uniprot_knowledgebase/ -t 5 -w 15 &&
	#/root/anaconda3/bin/python3 ProteomeAADistribution.py -d /home/vwesp/uniprot_knowledgebase/$domain/data -o /home/vwesp/uniprot_knowledgebase/$domain/aa_distributions -c 100 -t 20 &&
	#/root/anaconda3/bin/python3 GetGeneticEncoding.py -d /home/vwesp/uniprot_knowledgebase/$domain/aa_distributions/ -o /home/vwesp/uniprot_knowledgebase/$domain/encoding_data.csv &&
	/root/anaconda3/bin/python3 CombineAAStats.py -d /home/vwesp/uniprot_knowledgebase/$domain/aa_distributions/ -o /home/vwesp/uniprot_knowledgebase/$domain/combined_distributions.csv -e /home/vwesp/uniprot_knowledgebase/$domain/encoding_data.csv -c genetic_codes/ -m code_map.csv -r 100000 -ch 1000 -t 7
	echo ""
done

echo "Plotting statistics..."
/root/anaconda3/bin/python3 PlotAAStats.py -i /home/vwesp/uniprot_knowledgebase/ -o /home/vwesp/uniprot_knowledgebase/stats_results -r 100000
