# Projets de Data Science
  
Pour chaque projet, un fichier ReadMe explique l'objectif, les données utilisées et les fichiers compris dans le dossier.  
Les projets sont basés sur SQL et Python.  
Les librairies de base utilisées dans tous les projets (ou presque) sont numpy, pandas, sklearn et matplotlib.  
Les librairies joblib, logging et datetime sont également régulièrement utilisées.  
  
  
## Dossier data_analysis
Ce dossier contient mes projets du parcours Data Analyst d'OpenClassrooms en partenariat avec l'ENSAE-ENSAI. Tous les projets utilisent scipy et/ou statsmodels.  
- étude de santé publique basée sur les données de la FAO (DA03_Etude_Sante_Publique): sqlite3, sqlalchemy.
- analyse des ventes d'une entreprise - librairie en ligne (DA04_Analyse_Ventes).
- étude de marché pour cibler les pays propices à un développement à l'international (DA05_Etude_Marche): BeautifulSoup.
- détection de faux billets (DA06_Detection_Faux_Billets): Flask.
- prédiction de revenus pour cibler les clients potentiels d'une banque (DA07_Prediction_Revenus).
- communication de résultats (DA08_Communication_Resultats): rapport et dashboard Tableau.
- prédiction de la demande en électricité (DA09_Prediction_Consommation_Electrique): séries temporelles.  
  
## Dossier machine_learning
Ce dossier contient mes projets du parcours Ingénieur Machine Learning d'OpenClassrooms en partenariat avec CentraleSupélec.  
- analyse de données: étude sur l'alimentation basée sur les données d'Open Food Facts, pour trouver une idée d'application de santé publique (ML02_Etude_Sante_Publique): missingno, scipy, statsmodels.
- régression: estimation de la consommation électrique de bâtiments de Seattle (ML03_Consommation_Electrique_Batiments): dython, missingno, scipy, statsmodels, category_encoders, hyperopt, xgboost.
- segmentation des clients d'un site de e-commerce pour cibler les campagnes de communication (ML04_Segmentation_Clients): scipy.
- catégorisation des questions du site Stack Overflow pour suggérer automatiquement des tags (ML05_Categorisation_Questions): scikit-multilearn, nltk, BeautifulSoup, wordcloud, tmtoolkit, pyLDAvis, tensorflow_hub, hyperopt, xgboost.
- classification d'images de chiens et prédiction de leur race (ML06_Classification_Images): tensorflow, kerastuner, tensorboard.
- développement d'une preuve de concept sur le papier Learning What and Where To Transfer, basé sur la knowledge distillation (ML07_Proof_Concept): pytorch, tensorflow, tensorflow_hub, PIL.
- participation à une compétition Kaggle: prédiction des ventes futures de magasins (ML08_Predict_Future_Sales).  
  
Les modèles issus des projets 5 (catégorisation des questions Stack Overflow) et projets 6 (classification d'images de chiens) sont en production sur le site [friot.net/data](http://friot.net/data).  
