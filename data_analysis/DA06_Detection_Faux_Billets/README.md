# DA06_Detection_Faux_Billets
  
## Projet 6 du parcours Data Analyst : Détectez de faux billets
  
### La mission
  
Votre société de consulting informatique vous propose une nouvelle mission au ministère de l'intérieur, dans le cadre de la lutte contre la criminalité organisée, à l'office central pour la répression du faux monnayage . Votre mission si vous l'acceptez : créer un algorithme de détection de faux billets.  
Vous vous voyez déjà en grand justicier combattant sans relâche la criminalité organisée en pianotant à mains de maîtres votre ordinateur, pour façonner ce fabuleux algorithme qui traquera la moindre fraude et permettra de mettre à jour les réseaux secrets de fossoyeurs ! La classe, non ? ... bon, si on retombait les pieds sur terre? Travailler pour la police judiciaire, c'est bien, mais vous allez devoir faire appel à vos connaissances en statistiques, alors on y va !  
  
  
### Les données
  
La PJ vous transmet un jeu de données contenant les caractéristiques géométriques de billets de banque. Pour chacun d'eux, nous connaissons :
- La longueur du billet (en mm)
- La hauteur du billet (mesurée sur le côté gauche, en mm)
- La hauteur du billet (mesurée sur le côté droit, en mm)
- La marge entre le bord supérieur du billet et l'image de celui-ci (en mm)
- La marge entre le bord inférieur du billet et l'image de celui-ci (en mm)
- La diagonale du billet (en mm)
  
  
### Les fichiers du repo
  
**Notebook**  
Le notebook data_analyst_projet06.ipynb contient tout le code d'analyse et de modélisation du projet.  
A la fin du notebook, il y a une implémentation possible de la prédiction.  
Je fournis les fichiers traintest_billets_acp.csv et traintest_billets_brut.csv.  
Ils sont chargés automatiquement par le notebook s'ils existent afin d'assurer la reproductibilité des résultats.  
Les fichiers joblib et projet06_acp.csv sont fournis à titre informatif. Ils sont regénérés par le notebook lorsqu'il est exécuté.  
  
**API Flask de prédiction**  
J'ai également développé une API sous forme d'application Flask pour la prédiction. Elle possède deux modes d'entrée et sortie:
- une réception de données Json qui renvoie des données Json. Cette implémentation est testée dans le notebook data_analyst_projet06_apirest.ipynb ;
- une implémentation html sous forme de templates avec un formulaire de chargement d'un fichier csv et une sortie en html. Le chemin d'accès est localhost:5000/billets
Cette application Flask est dans le fichier : billets_api_rest.py  
Elle nécessite les dossiers templates et static.  
  
**Dossier donnees**  
Les fichiers notes.csv et soutenance_example.csv contiennent les fichiers de données.  
  
**Dossier modules**  
Modules personnels appelés par le notebook d'analyse:
- sf_stats_inferentielles.py
- sf_classification_acp.py
- sf_graphiques.py
- sf_modeles_regression.py
- sf_modeles_classif_supervisee.py
