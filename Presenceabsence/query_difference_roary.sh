#!/bin/bash
#for file in $(cat hi_list.txt) ; do ./6_runprokka_BGxx_v2.sh $file ./input/ASSEMBLY_fna_HI ./prokka_out_HI influenzae haemophilus ; done
#for file in $(cat mcat_list.txt) ; do ./6_runprokka_BGxx_v2.sh $file ./input/ASSEMBLY_fna_MCAT ./prokka_out_MCAT catarrhalis moraxella ; done
#SBATCH -N 1
##SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH -t 8-00:00
#SBATCH -J 7x
#MYPREF=$1
#MYPATH="./input/ASSEMBLY_fna/Sample_BG"
INPUT1="/home/lf481323/GENOMICS/2021_09_29_gono/prokka/prokka_out_linA/*/*.gff"
INPUT2="/home/lf481323/GENOMICS/2021_09_29_gono/prokka/prokka_out_linB/*/*.gff"
MYOUTP="./roary_out_difference"
#MYOUTP=
#MYOUTP=$3

mkdir -p $MYOUTP

unicy=roary

echo "Starting query pan genome difference"
query_pan_genome -a difference --input_set_one $INPUT1 --input_set_two $INPUT2 -g roary_out_419_06102021_1633532462/clustered_proteins
