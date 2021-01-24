# ML02_Etude_Sante_Publique
  
## Projet 2 du parcours d'Ingénieur Machine Learning : Concevez une application au service de la santé publique
  
### La mission
  
L'agence "Santé publique France" a lancé un appel à projets pour trouver des idées innovantes d’applications en lien avec l'alimentation. Vous souhaitez y participer et proposer une idée d’application.  
  
Extrait de l’appel à projets :  
Le jeu de données Open Food Fact est disponible sur le site officiel.  
Les champs sont séparés en quatre sections :
- Les informations générales sur la 􀁽che du produit : nom, date de modification, etc.
- Un ensemble de tags : catégorie du produit, localisation, origine, etc.
- Les ingrédients composant les produits et leurs additifs éventuels.
- Des informations nutritionnelles : quantité en grammes d’un nutriment pour 100 grammes du produit.
  
Après avoir lu l’appel à projets, voici les différentes étapes que vous avez identifiées :  
1) Traiter le jeu de données afin de repérer des variables pertinentes pour les traitements à venir. Automatiser ces traitements pour éviter de répéter ces opérations.  
Le programme doit fonctionner si la base de données est légèrement modifiée (ajout d’entrées, par exemple).  
2) Tout au long de l’analyse, produire des visualisations afin de mieux comprendre les données. Effectuer une analyse univariée pour chaque variable intéressante, afin de synthétiser son comportement.  
L’appel à projets spécifie que l’analyse doit être simple à comprendre pour un public néophyte. Soyez donc attentif à la lisibilité : taille des textes, choix des couleurs, netteté suf􀁽sante, et variez les graphiques (boxplots, histogrammes, diagrammes circulaires, nuages de points…) pour illustrer au mieux votre propos.  
3) Confirmer ou infirmer les hypothèses à l’aide d’une analyse multivariée. Effectuer les tests statistiques appropriés pour vérifier la significativité des résultats.  
4) Élaborer une idée d’application. Identifier des arguments justi􀁽ant la faisabilité (ou non) de l’application à partir des données Open Food Facts.  
5) Rédiger un rapport d’exploration et pitcher votre idée durant la soutenance du projet.  
  
  
### Les données
  
A sélectionner et télécharger sur le site https://world.openfoodfacts.org/
Les données initialement téléchargées ne sont pas disponibles sur le repo car le fichier csv pèse plus de 2 Go.
  
  
### Les fichiers du repo
  
Le repo contient :
- 2 notebooks avec le nettoyage des données d'une part, l'analyse et la proposition d'application d'autre part ;
- un dossier donnees qui comporte des fichiers csv avec des données nettoyées ;
- un dossier modules qui comprend des fichier py avec des modules personnels.  
