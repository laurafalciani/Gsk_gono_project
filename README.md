# Gsk_gono_project

-prokkaRoaryBlast- 
	prokka script: 6_runprokka_BGxx_v2.sh
	roary script: run_roary
	blast script: run_blast



-PresenceAbsence-
	-query_difference_roary.sh: script per trovare geni unici
	-SVMRFE.py: script per Recursive Features elimination con Support Vector Machine
	-scoary.sh: script per Scoary	   
-GNN-
	-DistMatrixCreation.py scipt per calcolare le matrici di distanza e statistiche dai file di allineamento
	-AdjacencyBinaryMatrix.py: script per binarizzare le matrici di distanza
	-mydata:script per creare il dataset/grafo per la rete neurale
	-trainSaveModel.py:script per trainare la rete e salvare il modello
	-test_accuracy.py: script per testare accuracy modello

-Silhouette-
	-Silhouette/Silhouette.py: script per calcolo score di Silhouette a partire dalle matrici output dello script 
	DistMatrixCreation.py 
