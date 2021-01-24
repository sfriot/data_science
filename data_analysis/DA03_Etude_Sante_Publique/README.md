# DA03_Etude_Sante_Publique

## Projet 3 du parcours Data Analyst : Réalisez une étude de santé publique

### La Mission  
  
Vous êtes intégré à une nouvelle équipe de chercheurs de la Food and Agriculture Organization of the United Nations (FAO), l'un des organes qui compose l'ONU et dont l'objectif est d' « aider à construire un monde libéré de la faim ».  
Votre équipe est chargée de réaliser une étude de grande ampleur sur le thème de la sous-nutrition dans le monde.  
Le problème de la faim est complexe et peut avoir de multiples causes, différentes selon les pays. L’étape préliminaire de cette étude sera donc d’établir un “état de l’art” des recherches déjà publiées, mais également de mener une étude statistique destinée à orienter les recherches vers des pays particuliers, et de mettre en lumière différentes causes de la faim. Ainsi, une poignée de data analysts (dont vous !) a été sélectionnée pour mener cette étape préliminaire. Lors de la première réunion, vous avez été désigné pour mettre une place la base de données que votre équipe pourra requéter (en SQL) à souhait pour réaliser cette étude statistique.  
  
  
### Les données
  
Les données sont disponibles sur le site de la FAO, à cet endroit : http://www.fao.org/faostat/fr/#data.  
En fonction des besoins exprimés plus bas dans cet énoncé, il vous appartiendra de choisir quelles données télécharger. Cependant, les rubriques qui vous seront utiles sont les suivantes :
- Bilans alimentaires
- Sécurité alimentaire, puis dans cette rubrique, Données de la sécurité alimentaire.
Cette rubrique contient de nombreux indicateurs, mais nous ne nous intéresserons ici qu’à l’indicateur Nombre de personnes sous-alimentées.  
  
  
### Les fichiers du repo
  
Le Notebook Jupyter comprend le nettoyage et l'analyse des données.  
Le fichier pdf résume les requêtes SQL exécutées pour certaines analyses.  
  
Le dossier "donnees" comprend les fichiers csv avec les données brutes.  
Le dossier "bdd_sqlite" comporte la base SQLite créée à partir des données dans mes dataframes Python, à l'aide de la librairie SQLAlchemy.  
