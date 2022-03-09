#!/bin/bash
#for file in $(cat hi_list.txt) ; do ./6_runprokka_BGxx_v2.sh $file ./input/ASSEMBLY_fna_HI ./prokka_out_HI influenzae haemophilus ; done
#for file in $(cat mcat_list.txt) ; do ./6_runprokka_BGxx_v2.sh $file ./input/ASSEMBLY_fna_MCAT ./prokka_out_MCAT catarrhalis moraxella ; done
#SBATCH -N 1
##SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH -t 8-00:00
#SBATCH -J 7x
MYPREF=$1
#MYPATH="./input/ASSEMBLY_fna/Sample_BG"
#MYPATH="./input/ASSEMBLY_fna"
MYPATH=$2
#MYOUTP="prokka_out"
MYOUTP=$3
MYSPEC=$4
MYGENU=$5
mkdir -p $MYOUTP

unicy=prokka

#for file in ASSEMBLY_fna/*.fna ; do file2=${file//_cleaned.fna/}; file3=${file2//ASSEMBLY_fna\/Sample_/} ; /projects/GENOMICS/APPLICATIONS/Python-2.7.9/bin/python2.7   /projects/GENOMICS/APPLICATIONS/rampart_dependencies/stats/quast-2.3/quast.py --output=quast/$file3 --threads=20 --est-ref-size=2000000  --gene-finding --min-contig 200 -l $file3 -R ../orig/Moraxella_catarrhalis_NCTC11020/GCA_900476075.1_49595_F01_genomic.fna -G ../orig/Moraxella_catarrhalis_NCTC11020/GCA_900476075.1_49595_F01_genomic.gff $file ; done

#for file in  $MYPATH$MYPREF*.fna ; do file2=${file//_cleaned.fna/}; file3=${file2//.\/input\/ASSEMBLY_fna\/Sample_/} ; if [ -d "./quast_out/$file3" ] ; then continue ; else mkdir "./quast_out/$file3" ; echo "Start $file3 assembly" ; sleep 5 ; srun -N 1 --cpus-per-task 8 -J $file3 $unicy --output=quast_out/$file3 --threads=8  --glimmer --min-contig 200 -l $file3 -r ../orig/Neisseria_gonorrhoeae_FA_1090/AE004969.fna -g ../orig/Neisseria_gonorrhoeae_FA_1090/AE004969.gff $file &
#echo "End $file3 assembly"; sleep 3 ; fi ; sleep 3 ; done

#for file in  $MYPATH$MYPREF*.fna ; do file2=${file//_cleaned.fna/}; file3=${file2//.\/input\/ASSEMBLY_fna\/Sample_/} ; echo $file; echo $file2; echo $file3; done




#prokka --outdir prokka_out/Sample_BG222  --addgenes --locustag BG222 --prefix BG222 --species 'gonorrhoeae' --genus 'neisseria' --usegenus --strain BG222 ./input/ASSEMBLY_fna/Sample_BG222_cleaned.fna  1>>prokka.log 2>>prokka.err



#for file in  $MYPATH\/$MYPREF\.fas ; do file2=${file//.fas/}; file3="pippo"; if [ -d "./$MYOUTP/$file3" ] ; then continue ; else echo "Start $file3 annotation" ; srun -N 1 --cpus-per-task 8 -J $file3 $unicy --outdir ./$MYOUTP/$file3  --cpus 4 --addgenes --locustag $file3 --prefix $file3 --species $MYSPEC  --genus $MYGENU --proteins --strain $file3 $file  1>>./$MYOUTP/$file3\_prokka.log 2>>./$MYOUTP/$file3\_prokka.err & echo "Submitted $file3 annotation"; sleep 2 ; fi ; sleep 2 ; done

for file in  $MYPATH\/$MYPREF\.fas ; do file2=${file//.fas/}; file3=${file2//$MYPATH\/sample_/}; file4=${file3%%_*}; if [ -d "./$MYOUTP/$file3" ] ; then continue ; else echo "Start $file3 annotation" ; srun -N 1 --cpus-per-task 8 -J $file3 $unicy --outdir ./$MYOUTP/$file3  --cpus 4 --addgenes --locustag "g$file4"  --prefix $file3 --species $MYSPEC  --genus $MYGENU --strain $file3 $file  1>>./$MYOUTP/$file3\_prokka.log 2>>./$MYOUTP/$file3\_prokka.err & echo "Submitted $file3 annotation"; sleep 2 ; fi ; sleep 2 ; done


#for file in  $MYPATH\/$MYPREF.fas ; do file2=${file//.fas/}; file3=${file2//$MYPATH\/sample_/};file4=${file3%%_*};echo "$file $file2 $file3 $file4";done 

