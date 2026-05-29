# F1 Strategy — Prédiction de Pit Stop avec le Machine Learning

> Projet pédagogique : prédire si un pilote de Formule 1 va s'arrêter aux stands au prochain tour, à partir de données de course réelles.

---

## Table des matières

1. [Le problème](#1-le-problème)
2. [Le dataset](#2-le-dataset)
3. [Concepts fondamentaux du ML](#3-concepts-fondamentaux-du-ml)
4. [Le code expliqué ligne par ligne](#4-le-code-expliqué-ligne-par-ligne)
5. [Les modèles utilisés](#5-les-modèles-utilisés)
6. [Les métriques d'évaluation](#6-les-métriques-dévaluation)
7. [Résultats obtenus](#7-résultats-obtenus)
8. [Glossaire technique](#8-glossaire-technique)

---

## 1. Le problème

En Formule 1, décider **quand rentrer aux stands** est l'une des décisions stratégiques les plus critiques d'une course.  
Ce projet entraîne un modèle de machine learning à répondre à cette question :

> **À partir de l'état actuel de la course (usure des pneus, position, temps au tour...), le pilote va-t-il s'arrêter au prochain tour ?**

C'est un problème de **classification binaire** : la réponse est soit `0` (non), soit `1` (oui).

---

## 2. Le dataset

**Fichier :** `f1_strategy_dataset_v4.csv`

| Caractéristique | Valeur |
|---|---|
| Nombre de lignes | 101 305 |
| Nombre de colonnes | 16 |
| Courses couvertes | 28 Grand Prix |
| Pilotes | 31 |

### Description des colonnes utilisées

| Colonne | Type | Description |
|---|---|---|
| `TyreLife` | Numérique | Nombre de tours effectués sur ce jeu de pneus |
| `Compound` | Texte | Type de pneu : SOFT, MEDIUM, HARD, INTERMEDIATE, WET |
| `Cumulative_Degradation` | Numérique | Perte de performance accumulée depuis le début du relais (en secondes) |
| `LapTime_Delta` | Numérique | Différence de temps au tour par rapport au tour précédent |
| `Position` | Numérique | Position du pilote sur la grille à ce tour |
| `RaceProgress` | Numérique | Proportion de la course écoulée (0.0 = départ, 1.0 = fin) |
| `Normalized_TyreLife` | Numérique | Usure du pneu normalisée entre 0 et 1 selon le composé |
| `Stint` | Numérique | Numéro du relais en cours (1 = premier relais, 2 = après le 1er arrêt...) |
| `PitNextLap` | **Cible (0/1)** | **1 si le pilote s'arrête au prochain tour, 0 sinon** |

---

## 3. Concepts fondamentaux du ML

### Qu'est-ce que le Machine Learning ?

Le machine learning est une branche de l'intelligence artificielle où l'on **ne programme pas les règles à la main** — on laisse le modèle les découvrir lui-même à partir de données.

Exemple classique :  
- Approche traditionnelle : `if TyreLife > 25 and Degradation > 4 then pit`  
- Approche ML : on donne des milliers d'exemples au modèle, il trouve lui-même les seuils et combinaisons optimales.

### Features (X) et Cible (y)

Toute tâche de ML supervisé repose sur cette distinction :

```
Features (X)                        Cible (y)
─────────────────────────────       ──────────────
TyreLife = 22                  →    PitNextLap = 1
Compound = MEDIUM
Degradation = -6.3s
LapTime_Delta = +0.4s
Position = 8
...
```

- **Features** : tout ce que le modèle "voit" pour faire sa prédiction
- **Cible** : ce qu'on veut prédire

### Apprentissage supervisé

Dans ce projet on utilise l'**apprentissage supervisé** : on dispose de données historiques où on connaît déjà la réponse (on sait si le pilote a effectivement pitté ou non). Le modèle apprend de ces exemples labellisés pour généraliser sur de nouveaux cas.

### Overfitting (surapprentissage)

L'overfitting se produit quand un modèle apprend les données par cœur au lieu de comprendre les tendances générales. Il performe très bien sur les données d'entraînement mais mal sur de nouvelles données.

```
Analogie : un étudiant qui mémorise les réponses d'examens passés
sans comprendre le cours — il échoue face à un nouvel examen.
```

C'est pour éviter ce problème qu'on sépare les données en train et test.

---

## 4. Le code expliqué ligne par ligne

### 4.1 Imports

```python
import pandas as pd
```
Pandas est la librairie de référence pour manipuler des tableaux de données en Python. `pd` est l'alias conventionnel.

```python
import numpy as np
```
NumPy fournit des opérations mathématiques rapides sur des tableaux. Utilisé en arrière-plan par sklearn et pandas.

```python
import matplotlib.pyplot as plt
import seaborn as sns
```
Librairies de visualisation. Matplotlib est le moteur de base, Seaborn ajoute des graphiques statistiques plus élaborés avec un style plus moderne.

```python
from sklearn.model_selection import train_test_split
```
Fonction qui divise automatiquement le dataset en jeu d'entraînement et de test.

```python
from sklearn.preprocessing import LabelEncoder
```
Outil pour convertir des valeurs texte (`SOFT`, `MEDIUM`...) en nombres entiers.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
```
Les deux algorithmes de classification qu'on va comparer.

```python
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)
```
Outils pour mesurer et visualiser la performance du modèle (voir section 6).

---

### 4.2 Chargement des données

```python
df = pd.read_csv('f1_strategy_dataset_v4.csv').dropna(subset=['Compound'])
```

- `pd.read_csv(...)` : lit le fichier CSV et le charge en mémoire sous forme de DataFrame (tableau)
- `.dropna(subset=['Compound'])` : supprime les lignes où la colonne `Compound` est vide (`NaN`). Sans ça, l'encodeur planterait sur des valeurs manquantes.

---

### 4.3 Encodage du composé

```python
le = LabelEncoder()
df['Compound_enc'] = le.fit_transform(df['Compound'])
```

Les algorithmes ML ne peuvent pas traiter du texte directement. On convertit donc :

```
HARD         → 0
INTERMEDIATE → 1
MEDIUM       → 2
SOFT         → 3
WET          → 4
```

- `LabelEncoder()` : crée l'encodeur
- `.fit_transform(...)` : apprend le mapping (`fit`) ET l'applique à la colonne (`transform`) en une seule opération
- Le résultat est stocké dans une nouvelle colonne `Compound_enc`

---

### 4.4 Définition des features et de la cible

```python
FEATURES = [
    'TyreLife',
    'Compound_enc',
    'Cumulative_Degradation',
    'LapTime_Delta',
    'Position',
    'RaceProgress',
    'Normalized_TyreLife',
    'Stint',
]
TARGET = 'PitNextLap'

X = df[FEATURES]
y = df[TARGET]
```

- `FEATURES` : liste des colonnes que le modèle utilisera comme entrées
- `TARGET` : la colonne à prédire
- `X` : le tableau de features (101 305 lignes × 8 colonnes)
- `y` : la série de labels (101 305 valeurs : 0 ou 1)

La convention `X` majuscule / `y` minuscule est universelle en ML Python.

---

### 4.5 Split train / test

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

| Paramètre | Valeur | Signification |
|---|---|---|
| `test_size=0.2` | 20% | 20% des données vont dans le jeu de test |
| `random_state=42` | 42 | Graine aléatoire fixe — garantit que le split est reproductible |
| `stratify=y` | y | Conserve la même proportion de pit stops (25%) dans train ET test |

Résultat :
- `X_train` / `y_train` : 81 044 lignes → le modèle apprend dessus
- `X_test` / `y_test` : 20 261 lignes → on évalue le modèle dessus, il ne les a jamais vues

---

### 4.6 Entraînement — Régression Logistique

```python
lr = LogisticRegression(max_iter=1000, random_state=42)
```
Crée le modèle. `max_iter=1000` : autorise jusqu'à 1000 itérations pour converger (la valeur par défaut de 100 peut être insuffisante sur des gros datasets).

```python
lr.fit(X_train, y_train)
```
**L'entraînement.** Le modèle analyse les 81 044 exemples et ajuste ses poids internes pour minimiser les erreurs. C'est ici que "l'apprentissage" se produit réellement.

```python
y_pred_lr = lr.predict(X_test)
```
**La prédiction.** Le modèle prédit `0` ou `1` pour chacun des 20 261 tours du jeu de test.

---

### 4.7 Entraînement — Random Forest

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
```

| Paramètre | Valeur | Signification |
|---|---|---|
| `n_estimators=100` | 100 | Nombre d'arbres dans la forêt |
| `random_state=42` | 42 | Reproductibilité |
| `n_jobs=-1` | -1 | Utilise tous les cœurs du processeur disponibles |

```python
rf.fit(X_train, y_train)
```
Entraîne les 100 arbres en parallèle. Plus long que la régression logistique, mais beaucoup plus puissant.

```python
y_pred_rf = rf.predict(X_test)
```
Chaque arbre vote. La classe majoritaire (pit ou pas pit) devient la prédiction finale.

---

### 4.8 Visualisations d'évaluation

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
```
Crée une figure avec 2 graphiques côte à côte (1 ligne, 2 colonnes), de taille 12×4 pouces.

```python
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), ...).plot(ax=ax)
```
- `confusion_matrix(y_test, y_pred)` : calcule la matrice de confusion (vrais/faux positifs et négatifs)
- `ConfusionMatrixDisplay(...).plot(ax=ax)` : l'affiche visuellement dans le graphique

```python
y_proba = model.predict_proba(X_test)[:, 1]
```
Au lieu d'un `0` ou `1` binaire, `.predict_proba()` retourne la **probabilité** d'appartenir à chaque classe.  
`[:, 1]` sélectionne uniquement la colonne de probabilité pour la classe `1` (pit stop).

```python
RocCurveDisplay.from_predictions(y_test, y_proba, ...)
```
Trace la courbe ROC en comparant les probabilités prédites avec les vraies valeurs.

---

### 4.9 Feature Importance

```python
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
```

- `rf.feature_importances_` : tableau de 8 valeurs (une par feature) calculées automatiquement par le Random Forest
- `pd.Series(..., index=FEATURES)` : transforme ce tableau en série avec les noms de colonnes comme index
- `.sort_values(ascending=True)` : trie du moins important au plus important (pour un graphique horizontal lisible)

```python
importances.plot(kind='barh', color='steelblue')
```
`kind='barh'` : graphique en barres horizontales (*bar horizontal*).

---

## 5. Les modèles utilisés

### Régression Logistique

Malgré son nom, c'est un algorithme de **classification**. Il calcule une combinaison linéaire des features et applique une fonction sigmoïde pour obtenir une probabilité.

```
P(pit) = 1 / (1 + e^(-(w1×TyreLife + w2×Compound + w3×Dégradation + ...)))
```

- Si `P(pit) > 0.5` → prédit `1`
- Si `P(pit) ≤ 0.5` → prédit `0`

**Limite principale ici :** suppose une relation *linéaire* entre features et cible. La décision de pitter en F1 est beaucoup plus complexe.

---

### Arbre de décision

Brique de base du Random Forest. Ressemble à un organigramme de questions/réponses :

```
TyreLife > 20 ?
├── OUI → Cumulative_Degradation > 5.0 ?
│          ├── OUI → PIT  ✓
│          └── NON → PAS DE PIT  ✗
└── NON → PAS DE PIT  ✗
```

Chaque nœud choisit la question qui sépare le mieux les deux classes.

---

### Random Forest (Forêt aléatoire)

Construit **100 arbres de décision différents**, chacun entraîné sur :
- Un sous-échantillon aléatoire de lignes (**bootstrap**)
- Un sous-ensemble aléatoire de features à chaque nœud

La prédiction finale = **vote majoritaire** des 100 arbres.

```
Arbre 1  → PIT
Arbre 2  → PIT
Arbre 3  → PAS DE PIT
...
Arbre 97 → PIT
Arbre 98 → PIT
─────────────────
Résultat → PIT  (majorité)
```

Ce mécanisme s'appelle le **bagging** (Bootstrap AGGregatING). La diversité forcée des arbres réduit l'overfitting.

---

## 6. Les métriques d'évaluation

### Accuracy (précision globale)

```
Accuracy = (bonnes prédictions) / (total prédictions)
```

**Piège :** si 75% des tours n'ont pas de pit stop, un modèle qui prédit toujours `0` obtient 75% d'accuracy sans rien apprendre. C'est pourquoi on utilise d'autres métriques.

---

### Matrice de confusion

|  | Prédit : 0 (pas de pit) | Prédit : 1 (pit) |
|---|---|---|
| **Réel : 0** | Vrai Négatif (TN) ✅ | Faux Positif (FP) ❌ |
| **Réel : 1** | Faux Négatif (FN) ❌ | Vrai Positif (TP) ✅ |

- **TN** : on prédit "pas de pit" → correct
- **TP** : on prédit "pit" → correct
- **FP** : fausse alarme (on prédit pit mais il ne rentre pas)
- **FN** : pit raté (on ne prédit pas pit mais il rentre quand même) → **le pire cas en F1**

---

### Precision

```
Precision = TP / (TP + FP)
```
Parmi tous les pit stops **prédits**, combien étaient **vraiment** des pit stops ?  
→ Mesure le taux de fausses alarmes.

---

### Recall (Sensibilité)

```
Recall = TP / (TP + FN)
```
Parmi tous les **vrais** pit stops, combien le modèle en a-t-il **détectés** ?  
→ Mesure le taux de manqués. En F1, un pit raté peut coûter la course.

---

### F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Moyenne harmonique entre precision et recall. Utile quand les classes sont déséquilibrées (comme ici : 75% / 25%).

---

### Courbe ROC & AUC

La **courbe ROC** (Receiver Operating Characteristic) trace le taux de vrais positifs vs le taux de faux positifs pour tous les seuils de décision possibles (pas seulement 0.5).

L'**AUC** (Area Under the Curve) résume la courbe en un seul nombre :

| AUC | Interprétation |
|---|---|
| 0.5 | Modèle aléatoire (inutile) |
| 0.7–0.8 | Acceptable |
| 0.8–0.9 | Bon |
| > 0.9 | Excellent |
| 1.0 | Parfait (suspect — overfitting probable) |

---

### Feature Importance (Impureté de Gini)

À chaque nœud d'un arbre, on choisit la feature qui réduit le plus le **désordre** dans les sous-groupes. Ce désordre est mesuré par l'**impureté de Gini** :

```
Gini = 1 - (p_pit² + p_no_pit²)
```

L'importance d'une feature = somme des réductions de Gini qu'elle a causées sur tous les arbres, normalisée à 1.

---

## 7. Résultats obtenus

### Régression Logistique

| Classe | Precision | Recall | F1 |
|---|---|---|---|
| Pas de pit | 0.78 | 0.95 | 0.86 |
| **Pit** | **0.59** | **0.21** | **0.31** |

**AUC : 0.755**

Conclusion : le modèle détecte seulement **21% des vrais pit stops**. Trop linéaire pour ce problème.

---

### Random Forest

| Classe | Precision | Recall | F1 |
|---|---|---|---|
| Pas de pit | 0.93 | 0.97 | 0.95 |
| **Pit** | **0.88** | **0.78** | **0.83** |

**AUC : 0.968**

Conclusion : excellent. Il détecte **78% des vrais pit stops** avec 88% de précision.

---

### Feature Importance

| Rang | Feature | Importance | Interprétation |
|---|---|---|---|
| 1 | `LapTime_Delta` | 19.4% | Perte de temps = signal fort de dégradation |
| 2 | `RaceProgress` | 16.9% | Les arrêts se concentrent à certains moments de la course |
| 3 | `Cumulative_Degradation` | 16.3% | Dégradation accumulée depuis le début du relais |
| 4 | `Normalized_TyreLife` | 14.1% | Usure relative selon le composé |
| 5 | `TyreLife` | 13.0% | Nombre de tours sur ce jeu de pneus |
| 6 | `Position` | 9.1% | La stratégie dépend de la position en course |
| 7 | `Stint` | 7.1% | Numéro de relais |
| 8 | `Compound_enc` | 4.0% | Le moins important (déjà capturé par les autres features) |

---

## 8. Glossaire technique

| Terme | Définition |
|---|---|
| **Algorithme** | Ensemble de règles qu'un modèle suit pour apprendre |
| **AUC** | Area Under the Curve — résumé de la courbe ROC entre 0 et 1 |
| **Bagging** | Technique qui entraîne plusieurs modèles sur des sous-échantillons aléatoires et vote |
| **Classification** | Prédire une catégorie (pit/pas de pit, spam/non-spam...) |
| **Classification binaire** | Classification avec exactement 2 classes (0 ou 1) |
| **DataFrame** | Tableau de données 2D dans pandas (lignes × colonnes) |
| **Dataset** | Jeu de données utilisé pour entraîner ou évaluer un modèle |
| **Encodage** | Conversion de valeurs texte en nombres pour les algorithmes ML |
| **F1-Score** | Métrique équilibrant precision et recall |
| **Feature** | Variable d'entrée utilisée par le modèle pour prédire |
| **Feature Importance** | Mesure de la contribution de chaque feature aux décisions du modèle |
| **Faux Négatif (FN)** | Le modèle prédit 0 mais la vraie réponse est 1 |
| **Faux Positif (FP)** | Le modèle prédit 1 mais la vraie réponse est 0 |
| **Hyperparamètre** | Paramètre de configuration du modèle fixé avant l'entraînement (ex: n_estimators) |
| **Impureté de Gini** | Mesure du désordre dans un nœud d'arbre de décision |
| **Label** | Valeur de la cible (0 ou 1) associée à un exemple |
| **LabelEncoder** | Outil sklearn convertissant des catégories texte en entiers |
| **Modèle** | Algorithme après entraînement, capable de faire des prédictions |
| **NaN** | Not a Number — valeur manquante dans un DataFrame |
| **n_estimators** | Nombre d'arbres dans un Random Forest |
| **Overfitting** | Modèle trop ajusté aux données d'entraînement, mauvais en généralisation |
| **Precision** | Part des prédictions positives qui sont correctes |
| **Predict** | Fonction qui applique un modèle entraîné sur de nouvelles données |
| **Recall** | Part des vrais positifs détectés par le modèle |
| **Régression** | Prédire une valeur numérique continue (ex: temps au tour) |
| **Régression Logistique** | Algorithme de classification malgré son nom, basé sur la sigmoïde |
| **ROC** | Courbe traçant vrais positifs vs faux positifs pour tous les seuils |
| **random_state** | Graine aléatoire pour rendre les résultats reproductibles |
| **Sigmoïde** | Fonction mathématique qui écrase n'importe quel nombre entre 0 et 1 |
| **sklearn** | Scikit-learn — librairie Python de référence pour le machine learning |
| **Stratify** | Option du split qui conserve les proportions de classes dans train et test |
| **Supervisé (apprentissage)** | Entraîner un modèle sur des données dont on connaît les réponses |
| **Test set** | Données mises de côté pour évaluer le modèle final |
| **Train set** | Données sur lesquelles le modèle apprend |
| **Train/test split** | Division du dataset en jeu d'entraînement et jeu d'évaluation |
| **Vrai Négatif (TN)** | Le modèle prédit 0 et la vraie réponse est bien 0 |
| **Vrai Positif (TP)** | Le modèle prédit 1 et la vraie réponse est bien 1 |

---

*Dataset : données de courses de Formule 1 — 28 Grand Prix, 31 pilotes.*  
*Librairies : pandas, numpy, matplotlib, seaborn, scikit-learn.*
