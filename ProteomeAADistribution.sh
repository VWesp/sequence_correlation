# Archaea tr
~/anaconda3/bin/python3 ProteomeAADistribution.py ~/mount/Archaea/data/ ~/mount/Archaea/abundances/tr/ tr 60 &&
# Archaea sp
~/anaconda3/bin/python3 ProteomeAADistribution.py ~/mount/Archaea/data/ ~/mount/Archaea/abundances/sp/ sp 60 &&

# Bacteria tr
~/anaconda3/bin/python3 ProteomeAADistribution.py ~/mount/Bacteria/data/ ~/mount/Bacteria/abundances/tr/ tr 60 &&
# Bacteria sp
~/anaconda3/bin/python3 ProteomeAADistribution.py ~/mount/Bacteria/data/ ~/mount/Bacteria/abundances/sp/ sp 60 &&

# Eukaryotes tr
~/anaconda3/bin/python3 ProteomeAADistribution.py ~/mount/Eukaryotes/data/ ~/mount/Eukaryotes/abundances/tr/ tr 60 &&
# Eukaryotes sp
~/anaconda3/bin/python3 ProteomeAADistribution.py ~/mount/Eukaryotes/data/ ~/mount/Eukaryotes/abundances/sp/ sp 60 &&

# Viruses tr
~/anaconda3/bin/python3 ProteomeAADistribution.py ~/mount/Viruses/data/ ~/mount/Viruses/abundances/tr/ tr 60 &&
# Viruses sp
~/anaconda3/bin/python3 ProteomeAADistribution.py ~/mount/Viruses/data/ ~/mount/Viruses/abundances/sp/ sp 60
