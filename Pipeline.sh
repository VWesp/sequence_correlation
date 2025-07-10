#!/bin/bash

for domain in "Archaea"; do
	echo $domain
	#/home/we63kel/anaconda3/bin/python3 DownloadProteome.py -d $domain -o /home/we63kel/uniprot_knowledgebase -t 5 -w 15 &&
	#/home/we63kel/anaconda3/bin/python3 ProteomeAADistribution.py -d /home/we63kel/uniprot_knowledgebase/$domain/data -o /home/we63kel/uniprot_knowledgebase/$domain/aa_distributions -c 1000 -t 60 &&
	#/home/we63kel/anaconda3/bin/python3 GetGeneticEncoding.py -d /home/we63kel/uniprot_knowledgebase/$domain/aa_distributions/ -o /home/we63kel/uniprot_knowledgebase/$domain/encoding_data.csv &&
	/home/we63kel/anaconda3/bin/python3 CombineAAStats.py -d /home/we63kel/uniprot_knowledgebase/$domain/aa_distributions/ -o /home/we63kel/uniprot_knowledgebase/$domain/combined_distributions.csv -e /home/we63kel/uniprot_knowledgebase/$domain/encoding_data.csv -c genetic_codes/ -m code_map.csv -r 10 -ch 1000 -t 60
	echo ""
done

echo "Plotting statistics..."
#/home/we63kel/anaconda3/bin/python3 PlotAAStats.py -i /home/we63kel/uniprot_knowledgebase/ -o /home/we63kel/uniprot_knowledgebase/stats_results -r 100000
