# F1 Strategy — Analyse de données avec PySpark

> Projet pédagogique : analyser les données de Formule 1 avec Apache Spark pour explorer la dégradation des pneus, les stratégies de course et le comportement des pilotes.

---

## Table des matières

1. [Pourquoi Spark ?](#1-pourquoi-spark-)
2. [Le dataset](#2-le-dataset)
3. [Concepts fondamentaux de Spark](#3-concepts-fondamentaux-de-spark)
4. [Le code expliqué ligne par ligne](#4-le-code-expliqué-ligne-par-ligne)
5. [Les analyses effectuées](#5-les-analyses-effectuées)
6. [Résultats obtenus](#6-résultats-obtenus)
7. [Glossaire technique](#7-glossaire-technique)

---

## 1. Pourquoi Spark ?

### Pandas vs Spark

| | Pandas | PySpark |
|---|---|---|
| **Données** | Tient en RAM (millions de lignes max) | Milliards de lignes, distribué sur plusieurs machines |
| **Exécution** | Sur une seule machine | Sur un cluster ou en local multi-cœurs |
| **Usage typique** | Analyse exploratoire, ML classique | Big Data, pipelines de production |
| **API** | DataFrame Python | DataFrame distribué + SQL |

Dans ce projet, les données tiennent en mémoire — on utilise Spark en mode **local** pour apprendre l'outil sur un cas concret, avec des données réelles.

### Le mode local

```python
.master('local[*]')
```
`local[*]` signifie : exécuter Spark sur **une seule machine**, en utilisant **tous les cœurs disponibles** (`*`). Pas besoin de cluster.

---

## 2. Le dataset

**Fichier :** `f1_strategy_dataset_v4.csv`

| Caractéristique | Valeur |
|---|---|
| Nombre de lignes | ~101 000 |
| Courses couvertes | 28 Grand Prix |
| Pilotes | 31 |

### Schéma des colonnes

| Colonne | Type Spark | Description |
|---|---|---|
| `Driver` | StringType | Code du pilote (VER, HAM, LEC...) |
| `LapNumber` | IntegerType | Numéro du tour dans la course |
| `Compound` | StringType | Type de pneu : SOFT, MEDIUM, HARD, INTERMEDIATE, WET |
| `Stint` | IntegerType | Numéro du relais (1 = premier jeu de pneus, 2 = après le 1er arrêt...) |
| `TyreLife` | DoubleType | Nombre de tours effectués sur ce jeu de pneus |
| `Position` | IntegerType | Position du pilote en course |
| `LapTime (s)` | DoubleType | Temps du tour en secondes |
| `Race` | StringType | Nom du Grand Prix |
| `Year` | IntegerType | Année de la course |
| `LapTime_Delta` | DoubleType | Différence de temps vs le pilote devant |
| `Cumulative_Degradation` | DoubleType | Dégradation totale accumulée depuis le début du relais (en secondes) |
| `PitStop` | IntegerType | 1 si le pilote s'est arrêté à ce tour, 0 sinon |
| `PitNextLap` | IntegerType | 1 si le pilote s'arrêtera au prochain tour, 0 sinon |
| `RaceProgress` | DoubleType | Fraction de course écoulée (0.0 = départ, 1.0 = fin) |
| `Normalized_TyreLife` | DoubleType | Usure du pneu normalisée entre 0 et 1 selon le composé |
| `Position_Change` | DoubleType | Places gagnées ou perdues par rapport au tour précédent |

---

## 3. Concepts fondamentaux de Spark

### Le DataFrame Spark

Comme un DataFrame Pandas, mais **distribué** : les données sont divisées en **partitions** réparties sur plusieurs processeurs (ou machines). Les opérations s'appliquent à toutes les partitions en parallèle.

```
DataFrame Spark
┌─────────────┬──────────────┬──────────────┐
│  Partition 1│  Partition 2 │  Partition 3 │
│  (CPU 1)    │  (CPU 2)     │  (CPU 3)     │
│  25k lignes │  25k lignes  │  25k lignes  │
└─────────────┴──────────────┴──────────────┘
```

### Évaluation paresseuse (Lazy Evaluation)

Spark ne fait **rien** tant qu'on ne lui demande pas explicitement un résultat. Chaque transformation (`.filter()`, `.groupBy()`...) construit un **plan d'exécution** sans déclencher de calcul.

Le calcul se déclenche uniquement sur une **action** : `.show()`, `.collect()`, `.toPandas()`, `.count()`...

```
Transformations (lazy)     Action (déclencheur)
─────────────────────── →  ──────────────────────
.filter(...)                .show()      ← calcul ici
.groupBy(...)               .collect()
.agg(...)                   .toPandas()
```

**Avantage :** Spark peut optimiser le plan entier avant d'exécuter, ce qui évite des calculs inutiles.

### Le Schema (typage explicite)

Contrairement à Pandas qui devine les types, Spark recommande de **déclarer le schéma à l'avance** :
- Plus rapide (pas d'inférence)
- Garanti contre les erreurs de type au chargement
- Nécessaire en production pour la robustesse

### Window Functions (Fonctions de fenêtre)

Les fonctions de fenêtre permettent de faire des calculs **en regardant les lignes voisines** d'une même partition logique, sans réduire le DataFrame.

```
Exemple : LapTime du tour précédent pour chaque pilote dans chaque course

Driver | LapNumber | LapTime | prev_LapTime
VER    | 1         | 99.5    | NULL          ← pas de tour précédent
VER    | 2         | 91.2    | 99.5
VER    | 3         | 90.8    | 91.2
HAM    | 1         | 98.1    | NULL          ← repart de zéro pour HAM
HAM    | 2         | 91.5    | 98.1
```

La fenêtre est définie par :
- `partitionBy` : groupes indépendants (pilote + course)
- `orderBy` : ordre de lecture dans chaque groupe (numéro de tour)

---

## 4. Le code expliqué ligne par ligne

### 4.1 Imports

```python
from pyspark.sql import SparkSession, functions as F, types as T
```
- `SparkSession` : point d'entrée unique pour toute session Spark
- `functions as F` : toutes les fonctions Spark (`F.mean()`, `F.col()`, `F.lag()`...) — alias `F` pour éviter les conflits avec les fonctions Python
- `types as T` : types de données Spark (`T.StringType()`, `T.IntegerType()`...)

```python
from pyspark.sql.window import Window
```
Classe nécessaire pour définir des fenêtres de calcul glissantes.

---

### 4.2 Création de la SparkSession

```python
spark = SparkSession.builder.appName('F1 Pipeline Notebook')\
    .master('local[*]')\
    .getOrCreate()
```

| Méthode | Rôle |
|---|---|
| `.builder` | Commence la configuration |
| `.appName('...')` | Nom affiché dans l'interface Spark UI |
| `.master('local[*]')` | Mode local, tous les cœurs disponibles |
| `.getOrCreate()` | Crée la session ou récupère celle existante |

---

### 4.3 Définition du schéma

```python
schema = T.StructType([
    T.StructField('Driver', T.StringType(), True),
    T.StructField('LapNumber', T.IntegerType(), True),
    ...
])
```

- `StructType` : conteneur de l'ensemble du schéma (comme une définition de table SQL)
- `StructField('nom', type, nullable)` : définit une colonne
  - `True` en 3ème argument = la colonne peut contenir des valeurs nulles

Types Spark utilisés :
| Type Spark | Équivalent Python | Exemple |
|---|---|---|
| `StringType()` | `str` | `"MEDIUM"` |
| `IntegerType()` | `int` | `42` |
| `DoubleType()` | `float` | `91.456` |

---

### 4.4 Chargement du CSV

```python
df = spark.read.csv(csv_path, header=True, schema=schema)\
    .withColumnRenamed('LapTime (s)', 'LapTime_s')
```

- `spark.read.csv(...)` : lit le fichier CSV en DataFrame distribué
- `header=True` : la première ligne contient les noms de colonnes
- `schema=schema` : applique le schéma défini plutôt que de le deviner
- `.withColumnRenamed('LapTime (s)', 'LapTime_s')` : renomme la colonne pour supprimer l'espace et les parenthèses, qui causent des problèmes dans les expressions

---

### 4.5 Nettoyage

```python
df = df.filter(F.col('Compound') != 'None')
```

- `F.col('Compound')` : référence à la colonne `Compound`
- `.filter(...)` : conserve uniquement les lignes qui vérifient la condition
- Supprime les lignes où `Compound` vaut la chaîne `'None'` (texte, pas une valeur nulle Python)

---

### 4.6 Window — Dernier tour par pilote

```python
w = Window.partitionBy('Driver').orderBy(F.col('LapNumber').desc())
```
Définit une fenêtre qui :
- Sépare les données par pilote (`partitionBy('Driver')`)
- Ordonne par numéro de tour décroissant (`.desc()`) — donc le tour le plus récent en premier

```python
one_row_per_driver = (
    df.filter(F.col('Driver').isin(driver_filter))
      .withColumn('rn', F.row_number().over(w))
      .filter(F.col('rn') == 1)
      .drop('rn')
)
```

- `.isin(driver_filter)` : conserve uniquement les lignes dont le Driver est dans la liste
- `F.row_number().over(w)` : numérote les lignes dans chaque fenêtre (1, 2, 3...) selon l'ordre défini
- `.filter(F.col('rn') == 1)` : garde uniquement le rang 1 = le dernier tour de chaque pilote
- `.drop('rn')` : supprime la colonne temporaire `rn` une fois utilisée

---

### 4.7 Meilleurs tours par pilote

```python
best_laps = df.groupBy('Driver')\
    .agg(F.min('LapTime_s').alias('BestLap'))\
    .orderBy('BestLap')
```

- `.groupBy('Driver')` : regroupe toutes les lignes d'un même pilote
- `.agg(...)` : applique une ou plusieurs fonctions d'agrégation sur chaque groupe
- `F.min('LapTime_s')` : calcule le minimum de `LapTime_s` dans le groupe
- `.alias('BestLap')` : renomme la colonne résultante
- `.orderBy('BestLap')` : trie par ordre croissant (meilleur temps en premier)

```python
best_laps_pdf = best_laps.toPandas()
```
`.toPandas()` est une **action** — déclenche le calcul Spark et ramène les données en mémoire Python sous forme de DataFrame Pandas. À utiliser uniquement quand les données sont petites (résultat agrégé).

---

### 4.8 Normalisation de RaceProgress

```python
df = df.withColumn('RaceProgress_norm',
    F.when(F.col('RaceProgress') > 1, F.col('RaceProgress') / 100.0)
     .otherwise(F.col('RaceProgress')))
```

- `.withColumn('nom', expression)` : ajoute ou remplace une colonne
- `F.when(condition, valeur_si_vrai).otherwise(valeur_si_faux)` : équivalent d'un `if/else` appliqué à chaque ligne
- Certaines lignes avaient `RaceProgress` entre 0 et 100 (pourcentage) au lieu de 0 et 1 — on normalise

---

### 4.9 Fonction LAG — Tour précédent

```python
w_2 = Window.partitionBy('Driver', 'Race', 'Year').orderBy('LapNumber')

df = df.withColumn('prev_LapTime', F.lag('LapTime_s').over(w_2))

df = df.withColumn(
    'Delta_LapTime_with_previous',
    F.col('LapTime_s') - F.col('prev_LapTime')
)
```

- `Window.partitionBy('Driver', 'Race', 'Year')` : chaque combinaison pilote/course/année est une fenêtre indépendante
- `F.lag('LapTime_s')` : pour chaque ligne, retourne la valeur de `LapTime_s` de la ligne **précédente** dans la fenêtre
  - Le premier tour de chaque pilote dans chaque course retourne `NULL` (pas de tour précédent)
- La soustraction `LapTime_s - prev_LapTime` donne le delta : positif = le pilote a ralenti, négatif = il a accéléré

---

### 4.10 Durée des stints

```python
w_stint = Window.partitionBy('Driver', 'Race', 'Year', 'Stint').orderBy('LapNumber')
df2 = df.withColumn('lap_in_stint', F.row_number().over(w_stint))
```
Numérote les tours au sein de chaque relais (stint) pour chaque pilote/course.

```python
stint_lengths = (
    df2.groupBy('Driver', 'Race', 'Year', 'Stint', 'Compound')
       .agg(F.max('lap_in_stint').alias('Stint_Laps'))
)
```
Le maximum du rang dans un stint = le nombre total de tours de ce stint.

```python
compound_stats = (
    stint_lengths.groupBy('Compound')
                 .agg(
                     F.mean('Stint_Laps').alias('avg_laps'),
                     F.expr('percentile_approx(Stint_Laps, 0.5)').alias('median_laps'),
                     F.expr('percentile_approx(Stint_Laps, array(0.25,0.75))').alias('iqr_quartiles'),
                     F.min('Stint_Laps').alias('min_laps'),
                     F.max('Stint_Laps').alias('max_laps'),
                     F.stddev('Stint_Laps').alias('stddev_laps'),
                     F.count('*').alias('num_stints')
                 )
)
```

| Fonction | Rôle |
|---|---|
| `F.mean()` | Moyenne arithmétique |
| `percentile_approx(..., 0.5)` | Médiane (50ème percentile) — calcul approché pour les gros volumes |
| `percentile_approx(..., array(0.25, 0.75))` | 1er et 3ème quartiles (IQR) |
| `F.min()` / `F.max()` | Valeurs extrêmes |
| `F.stddev()` | Écart-type (mesure de dispersion) |
| `F.count('*')` | Nombre de stints observés |

---

### 4.11 Analyse pilote × composé (Pivot)

```python
driver_compound_pivot = (
    driver_compound.filter(F.col('n_samples_compound') >= 5)
    .groupBy('Driver')
    .pivot('Compound')
    .agg(F.first('avg_delta_compound'))
    .orderBy(F.desc('Hard'))
)
```

- `.pivot('Compound')` : transforme les valeurs uniques de `Compound` en **colonnes** — c'est l'opération de pivot (longue → large)
- `F.first(...)` : prend la première valeur disponible après le pivot (une seule valeur par cellule ici)
- `F.desc('Hard')` : trie par dégradation HARD décroissante
- `NaN` dans le résultat : ce pilote n'a pas de données pour ce composé (filtre `>= 5` échantillons)

---

## 5. Les analyses effectuées

### Analyse 1 — Dernier tour de chaque pilote

**Question :** quel était l'état de chaque pilote à la fin de la course ?

**Technique :** Window function avec `row_number()` + filtre sur rang 1 après tri décroissant.

---

### Analyse 2 — Meilleurs tours par pilote

**Question :** quel est le meilleur temps au tour de chaque pilote sur l'ensemble du dataset ?

**Technique :** `groupBy` + `agg(F.min(...))` + `orderBy`.

---

### Analyse 3 — Delta moyen par composé

**Question :** quel type de pneu perd le plus de temps d'un tour à l'autre ?

**Technique :** calcul de `Delta_LapTime_with_previous` via `F.lag()`, puis agrégation par `Compound`.

**Interprétation :** un delta moyen négatif signifie que les pilotes s'améliorent en général (début de course rapide, warm-up). Plus le delta est proche de 0, plus le pneu est stable.

---

### Analyse 4 — Durée des stints par composé

**Question :** combien de tours dure en moyenne un relais selon le type de pneu ?

**Technique :** `row_number()` par stint + `max()` pour obtenir la longueur + stats complètes.

---

### Analyse 5 — Qui maltraite le plus ses pneus ?

**Question :** quels pilotes ont la plus grande dégradation moyenne sur leurs pneus ?

**Technique :** filtre pit stops exclus + `groupBy('Driver')` + `mean(Delta_LapTime_with_previous)`.

---

### Analyse 6 — Tableau pilote × composé

**Question :** pour chaque pilote, comment se compare sa dégradation selon le type de pneu ?

**Technique :** `pivot()` pour transformer les composés en colonnes.

---

## 6. Résultats obtenus

### Meilleurs tours (top 5)

| Pilote | Meilleur tour |
|---|---|
| VER | 67.012s |
| LEC | 67.583s |
| ALO | 67.694s |
| PIA | 67.924s |
| NOR | 68.016s |

---

### Delta moyen par composé

| Composé | Delta moyen (s) | Interprétation |
|---|---|---|
| WET | -3.76 | Forte amélioration (début de pluie, piste qui sèche) |
| INTERMEDIATE | -0.88 | Conditions changeantes |
| MEDIUM | -0.69 | Pneu de référence, stable |
| HARD | -0.63 | Le plus stable — dégradation la plus lente |
| SOFT | -0.51 | Paradoxe apparent (voir note ci-dessous) |

> **Note :** les SOFT montrent un delta plus faible en valeur absolue parce que leur durée de vie est courte — les pilotes les retirent avant la dégradation sévère. Le signal de dégradation est tronqué par les arrêts précoces.

---

### Durée des stints par composé

| Composé | Moy. tours | Médiane | Min | Max | Nb stints |
|---|---|---|---|---|---|
| WET | 8.3 | 4 | 1 | 21 | 48 |
| SOFT | 13.7 | 14 | 1 | 50 | 933 |
| INTERMEDIATE | 15.3 | 13 | 1 | 44 | 364 |
| MEDIUM | 18.7 | 18 | 1 | 76 | 2017 |
| HARD | 25.6 | 25 | 1 | 76 | 1755 |

Cohérent avec la physique F1 : les HARD durent 3× plus longtemps que les WET ou SOFT.

---

### Pilotes qui maltraitent le plus leurs pneus

Les pilotes avec le delta le plus **proche de 0** préservent mieux leurs pneus (moins de dégradation tour après tour) :

| Rang | Pilote | Delta moyen | Lecture |
|---|---|---|---|
| 1 (meilleur) | ANT | -0.18 | Préserve très bien les pneus |
| 2 | HAD | -0.24 | |
| ... | ... | ... | |
| Dernier | SAR | -2.02 | Dégradation la plus élevée |

> **Attention :** les pilotes avec un grand `stddev_delta` (HAM, SAI, NOR...) ont des **outliers importants** qui faussent la moyenne. Ce sont probablement des tours de safety car ou des incidents inclus dans les données.

---

## 7. Glossaire technique

| Terme | Définition |
|---|---|
| **Action** | Opération Spark qui déclenche réellement le calcul (`.show()`, `.collect()`, `.toPandas()`) |
| **Agrégation** | Calcul qui réduit un groupe de lignes à une seule valeur (mean, min, count...) |
| **alias()** | Renomme une colonne ou expression dans un DataFrame Spark |
| **agg()** | Méthode Spark pour appliquer des fonctions d'agrégation après un `groupBy()` |
| **Cluster** | Ensemble de machines travaillant ensemble pour exécuter Spark |
| **collect()** | Action Spark qui ramène toutes les données en mémoire Python — dangereux sur de gros volumes |
| **DataFrame** | Structure de données tabulaire (lignes × colonnes) — Spark ou Pandas |
| **desc()** | Tri décroissant dans Spark (`F.col('x').desc()` ou `.orderBy(F.desc('x'))`) |
| **DoubleType** | Type Spark pour les nombres décimaux (virgule flottante 64 bits) |
| **Évaluation paresseuse** | Spark ne calcule rien tant qu'une action n'est pas appelée — construit un plan d'abord |
| **F.col()** | Référence à une colonne par son nom dans une expression Spark |
| **F.lag()** | Fonction de fenêtre : retourne la valeur de la ligne précédente |
| **F.row_number()** | Fonction de fenêtre : attribue un rang entier unique à chaque ligne dans une fenêtre |
| **F.when().otherwise()** | Condition if/else appliquée ligne par ligne dans Spark |
| **filter()** | Conserve les lignes qui vérifient une condition (équivalent de WHERE en SQL) |
| **groupBy()** | Regroupe les lignes selon une ou plusieurs colonnes (équivalent GROUP BY en SQL) |
| **IQR** | Interquartile Range — écart entre le 1er et 3ème quartile, mesure de dispersion robuste |
| **IntegerType** | Type Spark pour les entiers (32 bits) |
| **isin()** | Vérifie si la valeur d'une colonne appartient à une liste |
| **isNotNull()** | Filtre les lignes où la valeur n'est pas nulle |
| **Lazy Evaluation** | Voir "Évaluation paresseuse" |
| **Médiane** | Valeur centrale d'une distribution — moins sensible aux valeurs extrêmes que la moyenne |
| **NaN** | Not a Number — valeur manquante (dans ce contexte : pas de données pour ce couple pilote/composé) |
| **orderBy()** | Trie le DataFrame par une ou plusieurs colonnes |
| **over()** | Applique une fonction de fenêtre sur une Window définie |
| **partitionBy()** | Divise les données en groupes indépendants pour les window functions |
| **percentile_approx()** | Calcule un percentile de manière approximative (efficace sur grands volumes) |
| **pivot()** | Transforme les valeurs d'une colonne en colonnes séparées (format long → large) |
| **Plan d'exécution** | Description de toutes les opérations que Spark va effectuer, optimisée avant exécution |
| **PySpark** | API Python pour Apache Spark |
| **Partition** | Sous-ensemble du DataFrame traité par un seul cœur/nœud en parallèle |
| **Schema** | Définition explicite des noms et types des colonnes d'un DataFrame |
| **Spark UI** | Interface web Spark pour visualiser les jobs, stages et tâches en cours |
| **SparkSession** | Point d'entrée unique pour interagir avec Spark (remplace SparkContext + SQLContext) |
| **stddev** | Écart-type — mesure de la dispersion autour de la moyenne |
| **StringType** | Type Spark pour les chaînes de caractères |
| **StructField** | Définition d'une colonne dans un StructType (nom, type, nullable) |
| **StructType** | Conteneur de StructField — représente le schéma complet d'un DataFrame |
| **Transformation** | Opération Spark qui crée un nouveau DataFrame sans déclencher de calcul (lazy) |
| **toPandas()** | Action Spark : convertit un Spark DataFrame en Pandas DataFrame (tout en RAM) |
| **Window** | Fenêtre de calcul glissante définie par partitionBy + orderBy |
| **withColumn()** | Ajoute ou remplace une colonne dans un DataFrame Spark |

---

*Dataset : données de courses de Formule 1 — 28 Grand Prix, 31 pilotes.*  
*Librairies : PySpark, pandas, matplotlib.*
