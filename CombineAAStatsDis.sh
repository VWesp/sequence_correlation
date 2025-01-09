#!/bin/bash

arr1=("archaea" "bacteria" "eukaryota" "viruses")
arr2=("tr" "sp")

for element in "${arr1[@]}"; do
  name="${element^}"
  echo "Current kingdom: $name"
  for type in "${arr2[@]}"; do
    echo "Current type: $type"
    python3 CombineAAStatsDis.py /media/valentin-wesp/Volume/uniprot_proteomes/"$element"/aa_count_results/"$type"/ genetic_codes/ cor_results/"$type"/"$element"/
  done
done
