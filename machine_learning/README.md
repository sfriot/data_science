# Data Scientist
  
Ce dossier contient mes projets du parcours Ingénieur Machine Learning d'OpenClassrooms en partenariat avec CentraleSupélec.  
  
Les librairies de base utilisées dans tous les projets (ou presque) sont numpy, pandas, sklearn et matplotlib.  
Les librairies joblib, logging et datetime sont également régulièrement utilisées.  
    
Les modèles issus des projets 5 (catégorisation des questions Stack Overflow) et projets 6 (classification d'images de chiens) sont en production sur le site [friot.net/data](http://friot.net/data).  
  
## Liste des projets
  
- analyse de données: étude sur l'alimentation basée sur les données d'Open Food Facts, pour trouver une idée d'application de santé publique (ML02_Etude_Sante_Publique): missingno, scipy, statsmodels.
- régression: estimation de la consommation électrique de bâtiments de Seattle (ML03_Consommation_Electrique_Batiments): dython, missingno, scipy, statsmodels, category_encoders, hyperopt, xgboost.
- segmentation des clients d'un site de e-commerce pour cibler les campagnes de communication (ML04_Segmentation_Clients): scipy.
- catégorisation des questions du site Stack Overflow pour suggérer automatiquement des tags (ML05_Categorisation_Questions): scikit-multilearn, nltk, BeautifulSoup, wordcloud, tmtoolkit, pyLDAvis, tensorflow_hub, hyperopt, xgboost.
- classification d'images de chiens et prédiction de leur race (ML06_Classification_Images): tensorflow, kerastuner, tensorboard.
- développement d'une preuve de concept sur le papier Learning What and Where To Transfer, basé sur la knowledge distillation (ML07_Proof_Concept): pytorch, tensorflow, tensorflow_hub, PIL.
- participation à une compétition Kaggle: prédiction des ventes futures de magasins (ML08_Predict_Future_Sales).  
  
Pour chaque projet, un fichier ReadMe explique l'objectif, les données utilisées et les fichiers compris dans le dossier.  
