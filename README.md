# Wisconsin Breast Cancer - Pipeline Machine Learning

## Description du projet

Ce projet applique un pipeline Machine Learning complet sur le dataset
Wisconsin Breast Cancer afin de prédire la variable cible `diagnosis`.

L'objectif est de comparer plusieurs modèles de classification supervisée et de
sélectionner le meilleur modèle selon trois métriques principales :

- Accuracy
- F1-score
- AUC-ROC

Le projet est conçu pour être exécuté sur Google Colab.

## Dataset

Le dataset utilisé est :

```text
dataset/WBCD_prognosis.csv
```

Il contient :

- 569 observations
- 32 colonnes
- une colonne identifiant : `id`
- une colonne cible : `diagnosis`
- 30 variables numériques décrivant les caractéristiques des cellules

La variable cible contient deux classes :

- `N` : classe négative, encodée 0
- `R` : classe positive, encodée 1

## Problème traité

Le problème est une classification binaire.

À partir des caractéristiques numériques des cellules, le modèle doit prédire si
une observation appartient à la classe `N` ou à la classe `R`.

Ce type de problème est important dans un contexte médical, car il faut obtenir
un modèle capable de bien distinguer les deux classes tout en limitant les
erreurs de classification.

## Pipeline Machine Learning suivi

Le projet respecte strictement le pipeline ML standard suivant :

1. Définir le problème
2. Collecter les données
3. Explorer les données avec EDA
4. Préparer les données
5. Choisir les modèles
6. Entraîner les modèles
7. Évaluer les modèles avec boucle de retour si performance insuffisante
8. Déployer le meilleur modèle
9. Monitorer les performances et les données

## Étapes détaillées

### 1. Définir le problème

Le projet vise à construire un modèle de classification binaire permettant de
prédire la classe `diagnosis`.

Les performances sont évaluées avec :

- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC

### 2. Collecter les données

Les données sont chargées à partir du fichier CSV avec `pandas`.

Dans Google Colab, si le fichier n'est pas trouvé automatiquement, le notebook
demande de le téléverser.

### 3. Explorer les données

L'EDA permet de comprendre la structure du dataset.

Le notebook affiche :

- les premières lignes du dataset
- les dimensions du dataset
- les types des variables
- les statistiques descriptives
- les valeurs manquantes
- la distribution des classes
- la matrice de corrélation

Cette étape permet aussi de vérifier s'il existe un déséquilibre entre les
classes.

### 4. Préparer les données

Les étapes de préparation sont :

- suppression de la colonne `id`
- séparation entre variables explicatives `X` et cible `y`
- encodage de la cible
- séparation train/test avec stratification
- détection du déséquilibre des classes
- test de différentes méthodes de prétraitement

Les techniques testées sont :

- `StandardScaler`
- `MinMaxScaler`
- sans PCA
- avec PCA conservant 95% de la variance

### 5. Choisir les modèles

Quatre modèles sont comparés :

- Logistic Regression
- KNN
- Decision Tree
- SVM

Ces modèles ont été choisis car ils représentent des approches différentes de la
classification supervisée.

## Justification du choix des modèles

### Logistic Regression

La régression logistique est utilisée comme modèle de référence.

Elle est adaptée à la classification binaire, rapide à entraîner et facilement
interprétable. Elle permet d'obtenir une première performance de base et
fonctionne bien lorsque les classes sont séparables de manière presque linéaire.

### KNN

KNN est utilisé pour tester une approche basée sur la similarité entre les
observations.

Il classe une nouvelle observation selon les classes de ses voisins les plus
proches. Il peut capturer des frontières de décision non linéaires, mais il est
sensible à l'échelle des variables, ce qui justifie l'utilisation de la
normalisation.

### Decision Tree

L'arbre de décision est choisi pour son interprétabilité.

Il construit des règles simples de décision et permet de comprendre quelles
variables influencent les prédictions. Il peut aussi capturer des relations non
linéaires entre les variables.

### SVM

SVM est utilisé car il est souvent performant sur des datasets médicaux de taille
moyenne avec plusieurs variables.

Il cherche une frontière optimale entre les classes. Avec un noyau non linéaire,
comme `rbf`, il peut modéliser des frontières de décision complexes.

## Gestion du déséquilibre des classes

Le notebook vérifie automatiquement la proportion de chaque classe dans le jeu
d'entraînement.

Si la classe minoritaire représente moins de 40% des données, un déséquilibre est
considéré comme présent.

Dans ce cas :

- Logistic Regression utilise `class_weight='balanced'`
- Decision Tree utilise `class_weight='balanced'`
- SVM utilise `class_weight='balanced'`
- KNN est évalué avec F1-score et AUC-ROC car il ne possède pas de paramètre
  `class_weight`

## Validation croisée

Le projet utilise une validation croisée k-fold stratifiée avec 5 folds :

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

La stratification permet de conserver la proportion des classes dans chaque fold.

Les modèles sont entraînés avec `GridSearchCV`, ce qui permet de tester plusieurs
combinaisons :

- hyperparamètres du modèle
- type de normalisation
- utilisation ou non de PCA

Le score principal utilisé pour sélectionner les meilleurs hyperparamètres est le
F1-score.

## Métriques d'évaluation

### Accuracy

L'accuracy mesure la proportion totale de prédictions correctes.

Elle est utile mais peut être trompeuse si les classes sont déséquilibrées.

### Precision

La precision mesure la proportion de prédictions positives qui sont réellement
positives.

### Recall

Le recall mesure la capacité du modèle à détecter les vrais positifs.

Dans un contexte médical, cette métrique est très importante, car manquer un cas
positif peut avoir des conséquences graves.

### F1-score

Le F1-score combine precision et recall.

Il est particulièrement utile lorsque les classes sont déséquilibrées.

### AUC-ROC

L'AUC-ROC mesure la capacité du modèle à séparer les deux classes.

Une AUC proche de 1 indique une très bonne séparation, tandis qu'une AUC proche
de 0.5 indique une performance proche du hasard.

## Visualisations générées

Le notebook génère plusieurs visualisations :

- distribution des classes
- matrice de corrélation
- matrices de confusion
- courbes ROC
- comparaison des métriques entre modèles
- importance des features

Ces visualisations permettent d'analyser les performances des modèles et
d'interpréter leurs décisions.

## Déploiement

Le meilleur modèle est sauvegardé avec `joblib`.

Le fichier généré est :

```text
/content/outputs/best_breast_cancer_model.joblib
```

Ce fichier contient le pipeline complet :

- prétraitement
- PCA si sélectionnée
- modèle entraîné

Il peut donc être réutilisé directement pour faire des prédictions sur de
nouvelles données ayant les mêmes colonnes explicatives.

## Monitoring

Le notebook propose une étape simple de monitoring.

Elle compare les moyennes des nouvelles données avec celles du jeu
d'entraînement afin de détecter une éventuelle dérive des données.

En production, il serait aussi recommandé de surveiller :

- la distribution des nouvelles données
- la proportion des classes prédites
- l'évolution de l'accuracy
- l'évolution du F1-score
- l'évolution de l'AUC-ROC
- la fréquence des erreurs de prédiction

## Fichiers du projet

```text
ML/
├── dataset/
│   └── WBCD_prognosis.csv
├── WBCD_ML_Pipeline_Colab.ipynb
├── breast_cancer_colab_pipeline.py
├── GUIDE_COLAB_ET_INTERPRETATION.md
├── requirements.txt
└── README.md
```

## Exécution sur Google Colab

### Option recommandée

1. Ouvrir Google Colab
2. Cliquer sur `File > Upload notebook`
3. Importer `WBCD_ML_Pipeline_Colab.ipynb`
4. Exécuter les cellules dans l'ordre
5. Téléverser `WBCD_prognosis.csv` si Colab le demande

### Installation des dépendances

Dans Colab, exécuter si nécessaire :

```python
!pip install -q pandas numpy scikit-learn matplotlib joblib
```

## Sorties générées

Après exécution, les résultats sont enregistrés dans :

```text
/content/outputs
```

Ce dossier contient :

- `class_distribution.png`
- `correlation_matrix.png`
- `cross_validation_results.csv`
- `test_results.csv`
- `confusion_matrices.png`
- `roc_curves.png`
- `model_comparison.png`
- `feature_importance.csv`
- `feature_importance.png`
- `best_breast_cancer_model.joblib`
- `monitoring_report.csv`
- `conclusion_rapport.txt`

## Interprétation attendue des résultats

Le meilleur modèle doit être sélectionné selon :

1. F1-score
2. AUC-ROC
3. Accuracy

Le F1-score et l'AUC-ROC sont prioritaires car le contexte médical nécessite un
modèle capable de bien détecter la classe positive tout en limitant les erreurs.

Une bonne performance est indiquée par :

- une accuracy élevée
- un F1-score élevé
- une AUC-ROC proche de 1
- peu de faux positifs et de faux négatifs dans la matrice de confusion

## Conclusion

Ce projet compare quatre modèles classiques de classification supervisée sur le
dataset Wisconsin Breast Cancer.

Les modèles sont évalués avec validation croisée, différentes techniques de
prétraitement et plusieurs métriques complémentaires.

Le modèle final est choisi selon ses performances sur le test set, puis sauvegardé
pour un déploiement simple.

Ce pipeline permet de respecter une démarche Machine Learning complète, depuis la
définition du problème jusqu'au monitoring du modèle.
