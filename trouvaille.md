# Analyse pneus F1 — Résumé & méthode

## Résumé
Notebook d'analyse PySpark exploratoire pour estimer :
- la durée typique d'un stint (en tours) par `Compound` (Soft/Medium/Hard),
- la distribution des tours restants avant pit (à partir de `PitNextLap`/`PitStop`),
- quels pilotes "malmènent" le plus leurs pneus (moyenne du delta de temps par rapport au tour précédent).

Les calculs sont réalisés dans `spark_analysis.ipynb` à partir du fichier `f1_strategy_dataset_v4.csv`.

## Fichiers clés
- `spark_analysis.ipynb` — notebook PySpark contenant le pipeline d'analyse.
- `f1_strategy_dataset_v4.csv` — dataset source.
- `pyspark-env/` — environnement virtuel local (optionnel).

# Méthodologie

## 1. Chargement des données

Le dataset est chargé avec un schéma explicite afin de garantir le bon typage des colonnes importantes telles que le pilote (`Driver`), le numéro de tour (`LapNumber`), le type de pneu (`Compound`), le stint (`Stint`) ainsi que les indicateurs liés aux arrêts au stand (`PitStop`, `PitNextLap`).

---

## 2. Nettoyage des données

Les lignes contenant une valeur de compound invalide (`Compound == 'None'`) sont supprimées afin de conserver uniquement les données exploitables pour l’analyse des stratégies pneumatiques.

---

## 3. Calcul des valeurs temporelles relatives

Une fenêtre est définie par pilote, course et année, puis triée selon le numéro du tour (`LapNumber`).

Cette fenêtre permet de récupérer le temps du tour précédent (`prev_LapTime`) afin de calculer la différence entre deux tours consécutifs (`Delta_LapTime_with_previous`).  
Cette métrique permet d’étudier l’évolution des performances tour après tour et d’observer les effets de la dégradation des pneus.

---

## 4. Calcul de la durée des stints

Pour chaque combinaison `(Driver, Race, Year, Stint)`, la position du tour à l’intérieur du stint est calculée.

La longueur totale d’un stint (`Stint_Laps`) correspond ensuite au nombre maximal de tours observés dans ce stint.  
Cette étape permet de mesurer la durée d’utilisation des différents types de pneus.

---

## 5. Statistiques par type de pneu

Des statistiques descriptives sont calculées pour chaque compound :

- moyenne ;
- médiane ;
- minimum ;
- maximum ;
- écart-type ;
- nombre total de stints observés.

Ces indicateurs permettent de comparer la longévité et la variabilité des différents types de pneus.

---

## 6. Validation via les événements de pit stop

Le nombre de tours restants avant un arrêt au stand (`laps_until_pit`) est estimé à partir de la position du tour dans le stint et de la longueur totale du stint.

Des agrégations par compound sont ensuite réalisées afin de vérifier la cohérence des comportements observés avec les événements de pit stop enregistrés dans les données.

---

## 7. Analyse de l’“abuse” des pneus par pilote

Les tours comportant un arrêt au stand ainsi que ceux ne possédant pas de tour précédent valide sont exclus de l’analyse.

Pour chaque pilote, plusieurs indicateurs sont ensuite calculés sur les variations de temps au tour :

- moyenne ;
- écart-type ;
- nombre d’échantillons.

Cette analyse permet d’évaluer la régularité des pilotes ainsi que leur capacité à préserver les performances des pneus au fil des tours.

---

## 8. Comparaison pilote × compound

Les résultats sont finalement pivotés par type de pneu (`Compound`) afin de comparer le comportement des pilotes selon les composés utilisés.

Cette étape permet d’identifier les différences de gestion pneumatique entre pilotes et d’observer l’impact des compounds sur la dégradation des performances.
