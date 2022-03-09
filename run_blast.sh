#!/bin/bash

#SBATCH -N 1
##SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH -t 8-00:00
#SBATCH -J 7x

DB="/hpc/scratch/hdd1/lf481323/UniProt/UP000000425.fasta"
input="/home/lf481323/GENOMICS/2021_09_29_gono/blast/input"


#makeblastdb -dbtype prot -in $DB

#for file in $input/*; do file1=$(basename $file); file2=${file1%%_*};
#echo "file $file2";
blastx -num_threads 4  -subject_besthit -query $input -db $DB -outfmt "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send sstrand evalue bitscore" > blast_report.txt


