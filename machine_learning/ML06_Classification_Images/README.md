# ML06_Classification_Images
  
## Projet 6 du parcours d'Ingénieur Machine Learning : Classez des images à l'aide d'algorithmes de Deep Learning
  
### La mission
  
Vous êtes bénévole pour l'association de protection des animaux de votre quartier. C'est d'ailleurs ainsi que vous avez trouvé votre compagnon idéal, Snooky. Vous vous demandez donc ce que vous pouvez faire en retour pour aider l'association.  
Vous apprenez, en discutant avec un bénévole, que leur base de données de pensionnaires commence à s'agrandir et qu'ils n'ont pas toujours le temps de référencer les images des animaux qu'ils ont accumulées depuis plusieurs années. Ils aimeraient donc obtenir un algorithme capable de classer les images en fonction de la race du chien présent sur l'image.  
  
L'association vous demande de réaliser un algorithme de détection de la race du chien sur une photo, afin d'accélérer leur travail d’indexation.  
Vous avez peu d’expérience sur le sujet vous décidez donc de contacter un ami expert en classification d’images. Il vous conseille dans un premier temps de pré-processer des images avec des techniques spécifiques (e.g. whitening, equalization, éventuellement modification de la taille des images) et de réaliser de la data augmentation (mirroring, cropping...).  
Ensuite, il vous incite à mettre en oeuvre deux approches s’appuyant sur l’état de l’art et l’utilisation de CNN (réseaux de neurones convolutionnels), que vous comparerez en termes de temps de traitement et de résultat :
1. Une première en réalisant votre propre réseau CNN, en vous inspirant de réseaux CNN existants. Prenez soin d'optimiser certains hyperparamètres (des layers du modèle, de la compilation du modèle et de l’exécution du modèle)
2. Une deuxième en utilisant le transfer learning, c’est-à-dire en utilisant un réseau déjà entraîné, et en le modifiant pour répondre à votre problème. Concernant le transfer learning, votre ami vous précise que :
- Une première chose obligatoire est de réentraîner les dernières couches pour prédire les classes qui vous intéressent seulement.
- Il est également possible d’adapter la structure (supprimer certaines couches, par exemple) ou de réentraîner le modèle avec un très faible learning rate pour ajuster les poids à votre problème (plus long) et optimiser les performances.
  
  
### Les données
  
Les bénévoles de l'association n'ont pas eu le temps de réunir les différentes images des pensionnaires dispersées sur leurs disques durs. Pas de problème, vous entraînerez votre algorithme en utilisant le Stanford Dogs Dataset : http://vision.stanford.edu/aditya86/ImageNetDogs/  
  
  
### Les fichiers du repo
  
**Notebooks**  
ML06_01_Tests.ipynb : préparation des données et essais de réseaux personnels  
ML06_02_TransferLearning.ipynb : modélisation par transfer learning  
  
  
**Programme de classification**  
Code permettant de classer une image, en ligne de commande :
- dog_race.py : fichier python contenant le code de classification
- dog_race.npy : sauvegarde de l'array Numpy avec les noms cleanés des races
- model : dossier contenant la sauvegarde du modèle utilisé pour la classification (tranfer learning avec fine-tuning basé sur EfficientNetB4)
- requirements.txt : fichier comportant les librairies requises, généré avec pip freeze
- conda_requirements.txt : fichier comportant les librairies requises, généré avec conda list
Ce code est basé sur Python 3.7.4 (Anaconda individual distribution). Les librairies requises, et leur version, sont indiquées dans le fichier requirements.txt.  
Attention, à l'heure de l'écriture de ces lignes, la version 2.3.0 de Tensorflow n'est pas disponible sur Conda. J'ai donc créé un environnement virtuel comprenant les autres librairies requises avec Conda, puis fait une installation manuelle de Tensorflow avec pip.  
  
  
**Autres dossiers**  
- weigths contient la sauvegarde des poids du modèle final.
- history_saves comporte des fichiers csv avec les sauvegardes des étapes des différents fitting effectués lors du projet.
- modules contient un module personnel d'affichage des graphiques  


### Amélioration dans le projet 7

Le repository ML07_Proof_Concept contient une amélioration de cette classification en ajoutant un détecteur d'objet qui permet de zoomer l'image sur le (ou les) chien(s) détecté(s) sur la photo.  
Les notebooks ML07bis_01_Create_IndexDirectory.ipynb, ML07bis_02_CreateCropImagesFolders.ipynb et ML07bis_03_EfficientNetB4_CropImages.ipynb contiennent ce travail.  
Le dernier notebook contient également une analyse de l'accuracy par race de chien.  

