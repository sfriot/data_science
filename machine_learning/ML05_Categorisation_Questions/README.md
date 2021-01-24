# ML05_Categorisation_Questions
  
## Projet 5 du parcours d'Ingénieur Machine Learning : Catégorisez automatiquement des questions
  
### La mission
  
Stack Overflow est un site célèbre de questions-réponses liées au développement informatique. Pour poser une question sur ce site, il faut entrer plusieurs tags de manière à retrouver facilement la question par la suite. Pour les utilisateurs expérimentés, cela ne pose pas de problème, mais pour les nouveaux utilisateurs, il serait judicieux de suggérer quelques tags relatifs à la question posée.  
Amateur de Stack Overflow, qui vous a souvent sauvé la mise, vous décidez d'aider la communauté en retour. Pour cela, vous développez un système de suggestion de tagpour le site. Celui-ci prendra la forme d’un algorithme de machine learning qui assigne automatiquement plusieurs tags pertinents à une question.  
  
Contraintes :
- Mettre en oeuvre une approche non supervisée.
- Utiliser une approche supervisée ou non pour extraire des tags à partir des résultats précédents.
- Comparer ses résultats à une approche purement supervisée, après avoir appliqué des méthodes d’extraction de features spécifiques des données textuelles.
- Mettre en place une méthode d’évaluation propre, avec une séparation du jeu de données pour l’évaluation.
- Pour suivre les modifications du code final à déployer, utiliser un logiciel de gestion de versions, par exemple Git.
  
  
### Les données
  
Stack Overflow propose un outil d’export de données - "stackexchange explorer", qui recense un grand nombre de données authentiques de la plateforme d’entraide.  
Par défaut, il y a une limite sur le temps d'exécution de chaque requête SQL, ce qui peut rendre difficile la récupération de toutes les données d'un coup. Pour récupérer plus de résultats, pensez à faire des requêtes avec des contraintes sur les id.  
  
  
### Les fichiers du repo
  
**Notebooks**  
ML05_01_Exploration.ipynb : préparation et exploration des données  
ML05_02_Unsupervised.ipynb : topic modeling  
ML05_03_Multilabel_Learning.ipynb : modèles de classification multi-label  
Les v2 sont des notebooks avec des ajouts pour enrichir le travail.  
  
**Dossier flask**  
Code de l'API Flask, qui suggère un ou plusieurs tags - voire aucune suggestion cohérente n'est trouvée - en fonction du titre et du texte de votre question.  
Ce code est basé sur Python 3.7.4 (Anaconda individual distribution). Les librairies requises, et leur version, sont indiquées dans le fichier requirements.txt.  
Pour tester localement, il faut utiliser les commandes en ligne suivantes :
1. Change directory to go on the directory where you have downloaded this flask folder, with the command cd
2. set FLASK_APP=sotags
3. set FLASK_DEBUG=0
4. flask run
Le premier chargement de l'application est un peu long à cause du téléchargement du modèle "Universal Sentence Encoder". Les chargements suivants sont plus rapides car les données d'Universal Sentence Encoder sont mises en cache.  
  
**Dossier rapport**  
Le fichier pdf est un rapport qui résume la recherche et ses conclusions.  
  
**Dossier donnees**  
Fournit les 10 fichiers csv avec les données téléchargées, ainsi qu'un fichier markdown qui indique les requêtes utilisées  
  
**Dossier resultats**  
Sauvegarde des résultats des optimisations avec hyperopt  

**Dossier modules**  
Modules personnels appelés par les notebooks  
