#!/bin/bash

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# WARNING! Running this script will permanently delete all .h5 files.
# They cannot be recovered from the bin.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

folders=(
"bowtie_Xiong_AuAu_wavleng_0.65_NEW_PARAMS"
"bowtie_Xiong_AuAu_wavleng_0.7"
"bowtie_Xiong_AuGaN_wavleng_0.65_gap_6"
)

for dir in "${folders[@]}"; do
    if [ -d "$dir" ]; then
        echo "Cleaning in: $dir"

        # rn this script olny shows which files will be deleted; for secure
        find "$dir" -type f -name "*.h5" -print # <- to delete sth change -print to -delete
    else
        echo "ERROR! The directory does not exist: $dir"
    fi
done

echo "DONE."
