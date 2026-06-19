# Guide Google Colab - Projet Wisconsin Breast Cancer

## 1. Ouvrir le notebook

Dans Google Colab :

1. Va sur https://colab.research.google.com
2. Clique sur `File > Upload notebook`
3. Choisis le fichier `WBCD_ML_Pipeline_Colab.ipynb`
4. Exécute les cellules dans l'ordre

Le notebook cherchera automatiquement le fichier `WBCD_prognosis.csv`.
S'il ne le trouve pas, Colab demandera de le téléverser.

## 2. Dataset utilisé

Le fichier détecté dans ce projet est :

```text
dataset/WBCD_prognosis.csv
```

Il contient :

- 569 observations
- 32 colonnes
- une colonne cible : `diagnosis`
- une colonne identifiant : `id`
- 30 variables numériques utilisées pour la prédiction

La cible contient deux classes :

- `N` : classe négative, codée 0
- `R` : classe positive, codée 1

## 3. Pipeline ML respecté

### 1. Définir le problème

Le problème est une classification binaire.
L'objectif est de prédire si une observation appartient à la classe `R` ou `N`
à partir des caractéristiques numériques du cancer du sein.

### 2. Collecter les données

Les données sont chargées avec `pandas.read_csv`.
Dans Colab, le notebook peut aussi demander l'upload du fichier CSV.

### 3. Explorer les données

Le notebook affiche :

- les premières lignes,
- les dimensions du dataset,
- les types des colonnes,
- les statistiques descriptives,
- les valeurs manquantes,
- la distribution des classes,
- une matrice de corrélation.

### 4. Préparer les données

Les actions réalisées sont :

- suppression de la colonne `id`,
- séparation entre variables `X` et cible `y`,
- encodage de `diagnosis`,
- séparation train/test stratifiée,
- détection du déséquilibre des classes,
- test de plusieurs prétraitements :
  - `StandardScaler`,
  - `MinMaxScaler`,
  - sans PCA,
  - avec PCA conservant 95% de la variance.

### 5. Choisir les modèles

Les quatre modèles demandés sont utilisés :

- Logistic Regression
- KNN
- Decision Tree
- SVM

Chaque modèle possède une grille d'hyperparamètres.

### 6. Entraîner les modèles

L'entraînement utilise :

- `Pipeline` de scikit-learn,
- `GridSearchCV`,
- validation croisée `StratifiedKFold` à 5 folds,
- optimisation selon le F1-score.

### 7. Évaluer les modèles

Les métriques utilisées sont :

- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC

Le meilleur modèle est choisi principalement selon :

1. F1-score
2. AUC-ROC
3. Accuracy

Le notebook contient aussi une boucle de retour logique :
si F1-score ou AUC-ROC sont insuffisants, il faut revenir aux étapes de préparation
et de choix des modèles.

### 8. Déployer

Le meilleur pipeline complet est sauvegardé dans :

```text
/content/outputs/best_breast_cancer_model.joblib
```

Ce fichier contient à la fois :

- le prétraitement,
- le modèle entraîné.

### 9. Monitorer

Le monitoring proposé compare les moyennes des nouvelles données avec celles
du jeu d'entraînement.
Cela permet de détecter une dérive simple des données.

## 4. Comment interpréter les performances

### Accuracy

L'accuracy mesure la proportion totale de prédictions correctes.
Elle est facile à comprendre, mais elle peut être trompeuse si les classes sont
déséquilibrées.

Exemple d'interprétation :

```text
Une accuracy de 0.96 signifie que 96% des observations du test set sont correctement classées.
```

### Precision

La precision mesure, parmi les observations prédites positives, combien sont réellement positives.

Exemple :

```text
Une precision élevée signifie que le modèle fait peu de fausses alertes positives.
```

### Recall

Le recall mesure, parmi les vrais positifs, combien sont détectés par le modèle.

Exemple :

```text
Un recall élevé signifie que le modèle manque peu de cas positifs.
```

Dans un contexte médical, le recall est souvent très important, car manquer un cas positif
peut être grave.

### F1-score

Le F1-score est la moyenne harmonique entre precision et recall.
Il est utile lorsque les classes sont déséquilibrées.

Exemple :

```text
Un F1-score élevé indique un bon équilibre entre fausses alertes et cas positifs manqués.
```

### AUC-ROC

L'AUC-ROC mesure la capacité du modèle à séparer les deux classes.

Exemple :

```text
Une AUC proche de 1 indique que le modèle distingue très bien les classes N et R.
Une AUC proche de 0.5 indique une performance proche du hasard.
```

## 5. Interprétation des modèles

### Logistic Regression

Avantages :

- simple,
- rapide,
- interprétable,
- souvent très performant sur ce dataset.

Limites :

- suppose une frontière de décision plutôt linéaire.

Interprétation possible :

```text
La régression logistique donne de bons résultats si les variables permettent une séparation presque linéaire entre les classes.
```

### KNN

Avantages :

- simple,
- non paramétrique,
- peut capturer des frontières non linéaires.

Limites :

- sensible au choix de `k`,
- sensible à l'échelle des variables,
- plus lent pour prédire sur de grands datasets.

Interprétation possible :

```text
KNN dépend fortement de la normalisation. C'est pourquoi StandardScaler et MinMaxScaler sont testés.
```

### Decision Tree

Avantages :

- facile à interpréter,
- capture des relations non linéaires,
- ne nécessite pas forcément de normalisation.

Limites :

- risque de surapprentissage si l'arbre est trop profond.

Interprétation possible :

```text
Un arbre de décision performant avec profondeur contrôlée indique que quelques règles peuvent bien séparer les classes.
```

### SVM

Avantages :

- très performant sur des datasets de taille moyenne,
- efficace avec séparation linéaire ou non linéaire,
- robuste avec un bon choix de noyau.

Limites :

- plus coûteux à entraîner,
- moins directement interprétable.

Interprétation possible :

```text
SVM obtient souvent de très bons scores si les classes sont bien séparables dans un espace transformé.
```

## 6. Comment commenter la matrice de confusion

La matrice contient :

- vrais négatifs : classe `N` correctement prédite,
- faux positifs : classe `N` prédite comme `R`,
- faux négatifs : classe `R` prédite comme `N`,
- vrais positifs : classe `R` correctement prédite.

Phrase utile :

```text
Le modèle est meilleur lorsque le nombre de faux positifs et de faux négatifs est faible.
Dans un contexte médical, les faux négatifs sont particulièrement importants à réduire.
```

## 7. Comment choisir le meilleur modèle

Tu peux écrire :

```text
Le meilleur modèle est choisi en comparant l'accuracy, le F1-score et l'AUC-ROC.
Comme le problème peut présenter un déséquilibre de classes et un enjeu médical,
le F1-score et l'AUC-ROC sont prioritaires par rapport à l'accuracy seule.
```

Puis :

```text
D'après les résultats obtenus, le modèle retenu est celui qui maximise le F1-score
et l'AUC-ROC sur le test set, tout en conservant une bonne accuracy.
```

## 8. Fichiers générés par le notebook

Dans `/content/outputs`, le notebook génère :

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

Ces fichiers peuvent être utilisés dans le rapport ou la présentation.

## 9. Conclusion type à adapter après exécution

```text
Dans ce projet, nous avons comparé quatre modèles de classification :
Logistic Regression, KNN, Decision Tree et SVM.
Chaque modèle a été évalué avec une validation croisée k-fold et testé avec
différentes techniques de prétraitement, notamment la normalisation et PCA.

Les performances ont été comparées avec l'accuracy, le F1-score et l'AUC-ROC.
Le meilleur modèle a été sélectionné selon ses performances sur le test set,
avec une attention particulière au F1-score et à l'AUC-ROC.

Les résultats montrent que le modèle retenu est capable de distinguer efficacement
les deux classes du dataset Wisconsin Breast Cancer.
```
