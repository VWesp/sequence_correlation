declare -a array=("human" "invertebrates" "mammals" "plants" "vertebrates")

for  i in ${!array[@]}; do
        wget -O ${array[$i]}/uniprot_sprot_${array[$i]}.dat.gz https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions/uniprot_sprot_${array[$i]}.dat.gz
        wget -O ${array[$i]}/uniprot_trembl_${array[$i]}.dat.gz https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions/uniprot_trembl_${array[$i]}.dat.gz
done
