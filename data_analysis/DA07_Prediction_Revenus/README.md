# DA07_Prediction_Revenus
  
## Projet 7 du parcours Data Analyst : Effectuez une prédiction de revenus
  
### La mission
  
Vous êtes employé dans une banque, présente dans de nombreux pays à travers le monde. Celle-ci souhaite cibler de nouveaux clients potentiels, plus particulièrement les jeunes en âge d'ouvrir leur tout premier compte bancaire. Cependant, elle souhaite cibler les prospects les plus susceptibles d'avoir, plus tard dans leur vie, de hauts revenus.  
L'équipe dans laquelle vous travaillez a donc reçu pour mission de créer un modèle permettant de déterminer le revenu potentiel d'une personne.  
  
"Quelles information avons-nous ?" demandez-vous à votre supérieur, qui vous répond : "A vrai dire... quasiment aucune : uniquement le revenu des parents, car nous allons cibler les enfants de nos clients actuels, ainsi que le pays où ils habitent. C'est tout ! Ah oui, une dernière chose : ce modèle doit être valable pour la plupart des pays du monde. Je vous laisse méditer là-dessus… Bon courage !"  
  
Avec aussi peu de données disponibles, cela semble être un sacré challenge !  
Ainsi, vous proposez une régression linéaire avec 3 variables :
- le revenu des parents,
- le revenu moyen du pays dans lequel habite le prospect,
- l'indice de Gini calculé sur les revenus des habitants du pays en question.
  
Ce projet ne traite que de la construction et de l'interprétation du modèle. Vous n'irez pas jusqu'à la phase de prédiction.  
  
  
### Les données
  
Ce fichier contient les données de la World Income Distribution, datée de 2008. Cette base de données est composée principalement d'études réalisées au niveau national pour bon nombre de pays, et contient les distributions de revenus des populations concernées.  
Vous téléchargerez également les indices de Gini estimés par la Banque Mondiale. Libre à vous de trouver également d'autres sources, ou de recalculer les indices de Gini à partir de la World Income Distribution.  
Vous aurez également besoin de récupérer le nombre d'habitants de chaque pays présent dans votre base.  
  
  
### Les fichiers du Repo
  
De nombreux Notebooks Jupyter montrent le traitement des données, les analyses initiales, l'échantillonnage et les différentes étapes (tentatives) de la modélisation.  
Le dossier donnees comprend toutes les données utilisées en entrée.  
Le dossier modules comporte les modules personnels appelés par les notebooks.
La sauvegarde issue de l'échantillonnage n'est pas disponible car le fichier était trop volumineux. Il faut le régénérer, si nécessaire.  
