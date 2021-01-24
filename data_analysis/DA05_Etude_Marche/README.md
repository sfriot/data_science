# DA05_Etude_Marche
  
## Projet 5 du parcours Data Analyst : Produisez une étude de marché
  
### La Mission  
  
Votre entreprise d'agroalimentaire souhaite se développer à l'international. Elle est spécialisée dans le poulet.  
L'international, oui, mais pour l'instant, le champ des possibles est bien large : aucun pays particulier ni aucun continent n'est pour le moment choisi. Tous les pays sont envisageables.  
Votre objectif sera d'aider à cibler plus particulièrement certains pays, dans le but d'approfondir ensuite l'étude de marché. Plus particulièrement, l'idéal serait de produire des "groupes" de pays, plus ou moins gros, dont on connaît les caractéristiques.  
Dans un premier temps, la stratégie est plutôt d'exporter les produits plutôt que de produire sur place, c'est-à-dire dans le(s) nouveau(x) pays ciblé(s).  
  
  
### Les données
  
Vous devez trouver les données sur le site de la FAO.  
  
  
### Les fichiers du repo
  
Le projet contient les fichiers suivants :  
  
1. Notebooks :
- data_analyst_projet05_getdistances.ipynb : scrapping des distances entre Paris et les capitales du monde ;
- data_analyst_projet05_preparation.ipynb : préparation des données. Importation, nettoyage et sélection ;
- data_analyst_projet05_analyse_base.ipynb : analyse initiale basée sur 5 variables ;
- data_analyst_projet05_tests_base.ipynb : test des résultats obtenus avec l'analyse de base ;
- data_analyst_projet05_analyse_complement.ipynb : analyse approfondie basée sur 11 variables ;
- data_analyst_projet05_tests_complement.ipynb : test des résultats obtenus avec l'analyse approfondie ;
- data_analyst_projet05_analyse_resume.ipynb : notebook comportant l'essentiel de la préparation des données, des analyses de base et approfondie, et des tests des résultats. Pour la présentation.
  
2. Dossier "csv_analyse" : fichiers csv :
- dendrogram_country_cluster_5variables.csv : liste des pays avec leur cluster (analyse de base)
- dendrogram_country_cluster_11variables.csv : liste des pays avec leur cluster (analyse approfondie)
- centroids_5variables.csv : liste des centroïdes avec leurs coordonnées dans les 5 dimensions (analyse de base)
- centroids_11variables.csv : liste des centroïdes avec leurs coordonnées dans les 11 dimensions (analyse approfondie)
- centroids_projected_5variables.csv : liste des centroïdes avec leurs coordonnées projetées sur les 4 premiers axes d'inertie (analyse de base)
- centroids_projected_11variables.csv : liste des centroïdes avec leurs coordonnées projetées sur les 8 premiers axes d'inertie (analyse approfondie)
- projet05_data_primaire_2013.csv : données conservées à la fin du notebook de préparation des données pour être utilisées dans les analyses. Données primaires utilisées pour les classifications et ACP. Données de 2013.
- projet05_data_secondaire_2013.csv : données conservées à la fin du notebook de préparation des données pour être utilisées dans les analyses. Données secondaires pour compléter l'analyse des résultats.  Données de 2013.
- projet05_data_primaire_mostrecent.csv : données conservées à la fin du notebook de préparation des données pour être utilisées dans les analyses. Données primaires utilisées pour les classifications et ACP. Données les plus récentes pour chaque variable.
- projet05_data_primaire_mostrecent.csv : données conservées à la fin du notebook de préparation des données pour être utilisées dans les analyses. Données secondaires pour compléter l'analyse des résultats.  Données les plus récentes pour chaque variable.
- projet05_liste_groupe_pays_FAO.csv : classement des pays par zone géographique
- 5variables_5clusters.csv : données sauvegardées à la fin des analyses pour être utilisées dans les tests.
- 5variables_6clusters.csv : données sauvegardées à la fin des analyses pour être utilisées dans les tests.
- 5variables_11clusters.csv : données sauvegardées à la fin des analyses pour être utilisées dans les tests.
- 11variables_8clusters.csv : données sauvegardées à la fin des analyses pour être utilisées dans les tests.  
  
3. Dossier "csv_donnees" : fichiers des données téléchargées
  
4. Dossier "dendogrammes" : deux fichiers png :
- dendrogramme_5variables.png : dendrogramme de l'analyse de base
- dendrogramme_11variables.png : dendrogramme de l'analyse approfondie  
  
5. Dossiers modules :
- sf_classification_acp.py : module python contenant mes fonctions d'analyse exploratoire (classification et ACP)
- sf_stats_inferentielles.py : : module python contenant mes fonctions de statistiques inférentielles  