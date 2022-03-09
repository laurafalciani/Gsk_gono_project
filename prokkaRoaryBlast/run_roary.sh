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
MYPATH="/home/lf481323/GENOMICS/2021_09_29_gono/prokka/prokka_out_all_06102021/*/*.gff"
#MYPATH=$2
MYOUTP="./roary_out_intermediate"
#MYOUTP=
#MYOUTP=$3

mkdir -p $MYOUTP

unicy=roary

echo "Starting roary for $MYPATH"
roary -f $MYOUTP -e -z -n -v -p 64 -r -s -ap $MYPATH 1>>./$MYOUTP/roary.log 2>>./$MYOUTP/.roary.err 
echo "Submitted pangenome analysis";

