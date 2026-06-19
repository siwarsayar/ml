# %% [markdown]
# # Pipeline ML standard - Wisconsin Breast Cancer Dataset
#
# Ce fichier est prêt pour Google Colab. Tu peux soit :
# - le téléverser dans Colab comme notebook/script,
# - soit copier les cellules dans un notebook `.ipynb`.
#
# Objectif : comparer Logistic Regression, KNN, Decision Tree et SVM sur le dataset
# Wisconsin Breast Cancer en respectant le pipeline ML standard.

# %% [markdown]
# ## 0. Installation et imports

# %%
# Sur Google Colab, exécute cette cellule telle quelle si une librairie manque.
# Les paquets sont souvent déjà installés dans Colab.
# !pip install -q pandas numpy scikit-learn matplotlib joblib

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE = 42
OUTPUT_DIR = Path("/content/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Définir le problème
#
# On veut construire un modèle de classification binaire pour prédire la classe
# `diagnosis` à partir des caractéristiques des cellules.
#
# Dans ce fichier :
# - `R` est traité comme classe positive, codée `1`
# - `N` est codé `0`
#
# Les modèles comparés sont :
# - Logistic Regression
# - KNN
# - Decision Tree
# - SVM
#
# Les critères de choix du meilleur modèle sont :
# - Accuracy
# - F1-score
# - AUC-ROC

# %%
TARGET_COLUMN = "diagnosis"
ID_COLUMN = "id"
POSITIVE_LABEL = "R"

# %% [markdown]
# ## 2. Collecter les données
#
# Dans Colab, commence par téléverser le fichier `WBCD_prognosis.csv` si tu ne l'as
# pas déjà placé dans `/content` ou `/content/dataset`.

# %%
def find_dataset() -> Path:
    possible_paths = [
        Path("/content/WBCD_prognosis.csv"),
        Path("/content/dataset/WBCD_prognosis.csv"),
        Path("dataset/WBCD_prognosis.csv"),
        Path("WBCD_prognosis.csv"),
    ]

    for path in possible_paths:
        if path.exists():
            return path

    csv_files = list(Path("/content").rglob("*.csv")) if Path("/content").exists() else []
    csv_files += list(Path(".").rglob("*.csv"))
    matching_files = [path for path in csv_files if "WBCD" in path.name or "breast" in path.name.lower()]
    if matching_files:
        return matching_files[0]

    try:
        from google.colab import files

        print("Aucun fichier CSV trouvé. Téléverse WBCD_prognosis.csv maintenant.")
        uploaded = files.upload()
        if not uploaded:
            raise FileNotFoundError("Aucun fichier téléversé.")
        uploaded_path = Path(next(iter(uploaded.keys())))
        if uploaded_path.suffix.lower() != ".csv":
            raise ValueError("Le fichier téléversé doit être un fichier CSV.")
        return uploaded_path
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(
            "Dataset introuvable. Place WBCD_prognosis.csv dans le dossier dataset/ "
            "ou dans le même dossier que ce script."
        ) from exc


data_path = find_dataset()
df = pd.read_csv(data_path)

print(f"Dataset chargé depuis : {data_path}")
print(f"Dimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes")
display(df.head())

# %% [markdown]
# ## 3. Explorer les données (EDA)
#
# Cette étape permet de comprendre :
# - les dimensions du dataset,
# - les types des colonnes,
# - les valeurs manquantes,
# - la distribution des classes,
# - les relations entre variables.

# %%
print("Informations générales :")
display(df.info())

print("\nStatistiques descriptives :")
display(df.describe())

print("\nValeurs manquantes par colonne :")
display(df.isna().sum())

print("\nDistribution de la cible :")
class_counts = df[TARGET_COLUMN].value_counts()
class_percentages = df[TARGET_COLUMN].value_counts(normalize=True).mul(100).round(2)
display(pd.DataFrame({"count": class_counts, "percentage": class_percentages}))

plt.figure(figsize=(6, 4))
class_counts.plot(kind="bar", color=["#4C78A8", "#F58518"])
plt.title("Distribution des classes")
plt.xlabel("Classe")
plt.ylabel("Nombre d'observations")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=160)
plt.show()

# %%
numeric_df = df.drop(columns=[ID_COLUMN], errors="ignore").copy()
numeric_df[TARGET_COLUMN] = numeric_df[TARGET_COLUMN].map({POSITIVE_LABEL: 1}).fillna(0)

plt.figure(figsize=(12, 10))
corr = numeric_df.corr(numeric_only=True)
plt.imshow(corr, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Corrélation")
plt.title("Matrice de corrélation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=160)
plt.show()

# %% [markdown]
# ## 4. Préparer les données
#
# Préparation appliquée :
# - suppression de `id`,
# - encodage de la cible,
# - séparation train/test stratifiée,
# - détection du déséquilibre de classes,
# - test de plusieurs prétraitements dans les pipelines :
#   - StandardScaler
#   - MinMaxScaler
#   - sans PCA
#   - PCA conservant 95% de la variance.

# %%
df_model = df.copy()

if ID_COLUMN in df_model.columns:
    df_model = df_model.drop(columns=[ID_COLUMN])

X = df_model.drop(columns=[TARGET_COLUMN])
y = (df_model[TARGET_COLUMN] == POSITIVE_LABEL).astype(int)
feature_names = X.columns.to_list()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

minority_ratio = y_train.value_counts(normalize=True).min()
use_balancing = minority_ratio < 0.40

print(f"Taille train : {X_train.shape}")
print(f"Taille test  : {X_test.shape}")
print(f"Ratio classe minoritaire dans train : {minority_ratio:.3f}")
print(f"Déséquilibre détecté : {'oui' if use_balancing else 'non'}")

# %% [markdown]
# ## 5. Choisir les modèles
#
# On définit les 4 modèles demandés et leurs grilles d'hyperparamètres.
#
# Pour gérer le déséquilibre :
# - Logistic Regression, Decision Tree et SVM utilisent `class_weight='balanced'`
#   lorsque la classe minoritaire est inférieure à 40%.
# - KNN ne possède pas `class_weight`, donc il est évalué avec F1 et AUC pour éviter
#   de choisir uniquement selon l'accuracy.

# %%
class_weight = "balanced" if use_balancing else None

models = {
    "Logistic Regression": {
        "estimator": LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE,
            class_weight=class_weight,
        ),
        "params": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["liblinear", "lbfgs"],
        },
    },
    "KNN": {
        "estimator": KNeighborsClassifier(),
        "params": {
            "model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"],
        },
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight=class_weight,
        ),
        "params": {
            "model__max_depth": [None, 3, 5, 7, 10],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
    },
    "SVM": {
        "estimator": SVC(
            probability=True,
            random_state=RANDOM_STATE,
            class_weight=class_weight,
        ),
        "params": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"],
        },
    },
}

preprocessing_grid = {
    "scaler": [StandardScaler(), MinMaxScaler()],
    "pca": ["passthrough", PCA(n_components=0.95, random_state=RANDOM_STATE)],
}

# %% [markdown]
# ## 6. Entraîner les modèles avec validation croisée k-fold
#
# On utilise `StratifiedKFold` pour garder la même proportion de classes dans chaque fold.
# La recherche d'hyperparamètres optimise le F1-score, tout en enregistrant aussi
# l'accuracy et l'AUC-ROC.

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "accuracy": "accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

searches = {}
cv_rows = []

for model_name, config in models.items():
    print(f"\nEntraînement : {model_name}")

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", "passthrough"),
            ("model", config["estimator"]),
        ]
    )

    param_grid = {
        **preprocessing_grid,
        **config["params"],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit="f1",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    searches[model_name] = search

    best_index = search.best_index_
    cv_rows.append(
        {
            "model": model_name,
            "best_cv_accuracy": search.cv_results_["mean_test_accuracy"][best_index],
            "best_cv_f1": search.cv_results_["mean_test_f1"][best_index],
            "best_cv_auc_roc": search.cv_results_["mean_test_roc_auc"][best_index],
            "best_params": search.best_params_,
        }
    )

cv_results = pd.DataFrame(cv_rows).sort_values(
    by=["best_cv_f1", "best_cv_auc_roc", "best_cv_accuracy"],
    ascending=False,
)

display(cv_results)
cv_results.to_csv(OUTPUT_DIR / "cross_validation_results.csv", index=False)

# %% [markdown]
# ## 7. Évaluer les modèles sur le test set
#
# On évalue chaque meilleur modèle trouvé par validation croisée.
# Si les performances sont insuffisantes, la boucle de retour consiste à revenir vers :
# - la préparation des données,
# - les hyperparamètres,
# - le choix du prétraitement,
# - la gestion du déséquilibre.
#
# Ici, le seuil minimal choisi est :
# - F1-score >= 0.90
# - AUC-ROC >= 0.90

# %%
MIN_F1 = 0.90
MIN_AUC = 0.90

test_rows = []
test_predictions = {}

for model_name, search in searches.items():
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    test_predictions[model_name] = {
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    test_rows.append(
        {
            "model": model_name,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
            "test_auc_roc": roc_auc_score(y_test, y_proba),
        }
    )

test_results = pd.DataFrame(test_rows).sort_values(
    by=["test_f1", "test_auc_roc", "test_accuracy"],
    ascending=False,
)

display(test_results)
test_results.to_csv(OUTPUT_DIR / "test_results.csv", index=False)

best_model_name = test_results.iloc[0]["model"]
best_model = searches[best_model_name].best_estimator_

print(f"Meilleur modèle : {best_model_name}")
print("Meilleurs paramètres :")
display(searches[best_model_name].best_params_)

if (
    test_results.iloc[0]["test_f1"] < MIN_F1
    or test_results.iloc[0]["test_auc_roc"] < MIN_AUC
):
    print(
        "Performance insuffisante : retourner à l'étape 4/5 pour tester "
        "d'autres prétraitements, plus d'hyperparamètres ou un rééquilibrage."
    )
else:
    print("Performance satisfaisante : on peut passer au déploiement.")

# %%
for model_name, preds in test_predictions.items():
    print(f"\nRapport de classification : {model_name}")
    print(classification_report(y_test, preds["y_pred"], target_names=["N", "R"]))

# %% [markdown]
# ### Matrices de confusion

# %%
n_models = len(test_predictions)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for ax, (model_name, preds) in zip(axes, test_predictions.items()):
    cm = confusion_matrix(y_test, preds["y_pred"])
    ConfusionMatrixDisplay(cm, display_labels=["N", "R"]).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )
    ax.set_title(model_name)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=160)
plt.show()

# %% [markdown]
# ### Courbes ROC

# %%
plt.figure(figsize=(8, 6))
ax = plt.gca()

for model_name, preds in test_predictions.items():
    auc = roc_auc_score(y_test, preds["y_proba"])
    RocCurveDisplay.from_predictions(
        y_test,
        preds["y_proba"],
        name=f"{model_name} (AUC={auc:.3f})",
        ax=ax,
    )

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("Courbes ROC des modèles")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=160)
plt.show()

# %% [markdown]
# ### Comparaison visuelle des métriques

# %%
metrics_to_plot = ["test_accuracy", "test_f1", "test_auc_roc"]
plot_df = test_results.set_index("model")[metrics_to_plot]

plot_df.plot(kind="bar", figsize=(9, 5), ylim=(0, 1), color=["#4C78A8", "#F58518", "#54A24B"])
plt.title("Comparaison des performances sur le test set")
plt.ylabel("Score")
plt.xlabel("Modèle")
plt.xticks(rotation=20, ha="right")
plt.legend(["Accuracy", "F1-score", "AUC-ROC"])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=160)
plt.show()

# %% [markdown]
# ### Importance des features
#
# Pour comparer l'importance des variables de façon compatible avec tous les modèles,
# on utilise `permutation_importance` sur le meilleur modèle.

# %%
perm = permutation_importance(
    best_model,
    X_test,
    y_test,
    n_repeats=20,
    random_state=RANDOM_STATE,
    scoring="f1",
    n_jobs=-1,
)

importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }
).sort_values(by="importance_mean", ascending=False)

display(importance_df.head(15))
importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

top_features = importance_df.head(15).sort_values("importance_mean")
plt.figure(figsize=(9, 6))
plt.barh(top_features["feature"], top_features["importance_mean"], color="#4C78A8")
plt.xlabel("Baisse moyenne du F1-score après permutation")
plt.title(f"Importance des features - {best_model_name}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160)
plt.show()

# %% [markdown]
# ## 8. Déployer
#
# Pour un projet académique, un déploiement simple consiste à sauvegarder :
# - le meilleur pipeline complet,
# - les métriques,
# - les figures.
#
# Le pipeline sauvegardé contient le prétraitement et le modèle, donc il peut recevoir
# directement les mêmes colonnes que `X`.

# %%
model_path = OUTPUT_DIR / "best_breast_cancer_model.joblib"
joblib.dump(best_model, model_path)

print(f"Modèle sauvegardé dans : {model_path}")

# Exemple d'utilisation du modèle sauvegardé
loaded_model = joblib.load(model_path)
sample_prediction = loaded_model.predict(X_test.iloc[[0]])[0]
sample_probability = loaded_model.predict_proba(X_test.iloc[[0]])[0, 1]

print(f"Exemple prédiction : {'R' if sample_prediction == 1 else 'N'}")
print(f"Probabilité classe R : {sample_probability:.3f}")

# %% [markdown]
# ## 9. Monitorer
#
# En production, il faut surveiller :
# - la distribution des nouvelles données,
# - la proportion des classes prédites,
# - l'évolution de l'accuracy, du F1-score et de l'AUC-ROC si les vraies classes sont disponibles,
# - les dérives de données par rapport au train set.
#
# Exemple simple : comparer les moyennes des nouvelles données avec celles du train set.

# %%
def monitor_new_data(new_data: pd.DataFrame, reference_data: pd.DataFrame) -> pd.DataFrame:
    """Compare les moyennes des nouvelles données avec celles du train set."""
    monitoring = pd.DataFrame(
        {
            "train_mean": reference_data.mean(numeric_only=True),
            "new_data_mean": new_data.mean(numeric_only=True),
        }
    )
    monitoring["absolute_difference"] = (
        monitoring["new_data_mean"] - monitoring["train_mean"]
    ).abs()
    return monitoring.sort_values("absolute_difference", ascending=False)


# Simulation : on surveille ici le test set comme s'il s'agissait de nouvelles données.
monitoring_report = monitor_new_data(X_test, X_train)
display(monitoring_report.head(10))
monitoring_report.to_csv(OUTPUT_DIR / "monitoring_report.csv")

print(f"Tous les résultats sont dans : {OUTPUT_DIR}")

# %% [markdown]
# ## Conclusion automatique
#
# Cette cellule génère un résumé textuel à intégrer dans le rapport.

# %%
best_row = test_results.iloc[0]

print("Conclusion automatique")
print("=" * 24)
print(
    f"Le meilleur modèle est {best_row['model']} avec une accuracy de "
    f"{best_row['test_accuracy']:.3f}, un F1-score de {best_row['test_f1']:.3f} "
    f"et une AUC-ROC de {best_row['test_auc_roc']:.3f}."
)
print(
    "Le choix du meilleur modèle se base principalement sur le F1-score et "
    "l'AUC-ROC, car ces métriques sont plus informatives que l'accuracy seule "
    "lorsqu'il existe un déséquilibre entre les classes."
)

print("\nClassement final des modèles :")
for rank, (_, row) in enumerate(test_results.iterrows(), start=1):
    print(
        f"{rank}. {row['model']} - Accuracy={row['test_accuracy']:.3f}, "
        f"F1={row['test_f1']:.3f}, AUC-ROC={row['test_auc_roc']:.3f}"
    )

print("\nInterprétation :")
if best_row["test_f1"] >= 0.90 and best_row["test_auc_roc"] >= 0.90:
    print(
        "Les performances sont satisfaisantes. Le modèle distingue correctement "
        "les classes N et R et peut être sauvegardé pour un déploiement simple."
    )
else:
    print(
        "Les performances ne sont pas encore suffisantes. Il faut revenir aux "
        "étapes de préparation des données, de choix des modèles ou de réglage "
        "des hyperparamètres."
    )

conclusion_text = f"""Conclusion du projet

Le meilleur modèle obtenu est {best_row['model']}.
Sur le test set, il obtient :
- Accuracy : {best_row['test_accuracy']:.3f}
- F1-score : {best_row['test_f1']:.3f}
- AUC-ROC : {best_row['test_auc_roc']:.3f}

Le modèle est sélectionné en priorité selon le F1-score et l'AUC-ROC, car le
contexte médical exige de bien détecter la classe positive tout en limitant les
erreurs de classification. L'accuracy est également prise en compte, mais elle
ne suffit pas seule pour juger la qualité du modèle.
"""

(OUTPUT_DIR / "conclusion_rapport.txt").write_text(conclusion_text, encoding="utf-8")
print(f"\nConclusion sauvegardée dans : {OUTPUT_DIR / 'conclusion_rapport.txt'}")

# %% [markdown]
# ## Interprétation attendue
#
# Après exécution :
# - Le tableau `cross_validation_results` indique la performance moyenne en k-fold.
# - Le tableau `test_results` indique la performance finale sur données jamais vues.
# - Le meilleur modèle est celui qui maximise surtout F1-score et AUC-ROC, puis accuracy.
# - La matrice de confusion montre les erreurs classe par classe.
# - La courbe ROC montre la capacité du modèle à séparer les classes.
# - L'importance des features montre quelles variables influencent le plus la prédiction.
