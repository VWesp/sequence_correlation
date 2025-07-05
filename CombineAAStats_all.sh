for domain in "Archaea" "Bacteria" "Eukaryota" "Viruses"; do
	echo $domain
	/root/anaconda3/bin/python3 GetGeneticEncoding.py -d /home/vwesp/uniprot_knowledgebase/$domain/aa_distributions/ -o /home/vwesp/uniprot_knowledgebase/$domain/encoding_data.csv &&
	/root/anaconda3/bin/python3 CombineAAStats.py -d /home/vwesp/uniprot_knowledgebase/$domain/aa_distributions/ -o /home/vwesp/uniprot_knowledgebase/$domain/ -e /home/vwesp/uniprot_knowledgebase/$domain/encoding_data.csv -c genetic_codes/ -m code_map.csv -r 10000 -ch 1000 -t 20
	echo ""
done
