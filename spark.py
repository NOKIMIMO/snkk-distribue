from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window
import pandas as pd
import matplotlib.pyplot as plt

#LOAD info
spark = SparkSession.builder.appName('F1 Pipeline Notebook')\
    .master('local[*]')\
    .getOrCreate()

# Column name + typing
schema = T.StructType([
    T.StructField('Driver', T.StringType(), True),                   # Nom du pilote # Identifier
    T.StructField('LapNumber', T.IntegerType(), True),               # Numéro du tour # Identifier
    T.StructField('Compound', T.StringType(), True),                 # Type de pneu (Soft, Medium, Hard)
    T.StructField('Stint', T.IntegerType(), True),                   # Numéro de stint (séquence de tours avec le même pneu)
    T.StructField('TyreLife', T.DoubleType(), True),                 # Durée de vie du pneu
    T.StructField('Position', T.IntegerType(), True),                # Position du pilote
    T.StructField('LapTime (s)', T.DoubleType(), True),              # Temps du tour (en secondes)
    T.StructField('Race', T.StringType(), True),                     # Nom de la course # Identifier
    T.StructField('Year', T.IntegerType(), True),                    # Année de la course # Identifier
    T.StructField('LapTime_Delta', T.DoubleType(), True),            # Différence de temps par rapport au pilote devant
    T.StructField('Cumulative_Degradation', T.DoubleType(), True),   # Dégradation cumulative du pneu
    T.StructField('PitStop', T.IntegerType(), True),                 # Indicateur de pit stop (1 si le pilote a effectué un pit stop à ce tour, 0 sinon)
    T.StructField('PitNextLap', T.IntegerType(), True),              # Indicateur de pit stop au tour suivant (1 si le pilote effectuera un pit stop au tour suivant, 0 sinon)
    T.StructField('RaceProgress', T.DoubleType(), True),             # Progression de la course (en pourcentage)
    T.StructField('Normalized_TyreLife', T.DoubleType(), True),      # Durée de vie du pneu normalisée par rapport à la durée de vie maximale observée pour ce type de pneu
    T.StructField('Position_Change', T.DoubleType(), True),          # Changement de position par rapport au tour précédent (en nombre de places gagnées ou perdues)
])

csv_path = 'f1_strategy_dataset_v4.csv'
df = spark.read.csv(csv_path, header=True, schema=schema)\
    .withColumnRenamed('LapTime (s)', 'LapTime_s')

# header rapide 
# utilisé des drivers distincts
driver_filter = [row['Driver'] for row in df.select('Driver').distinct().collect()][:5]
print("Drivers présents dans le dataset :", driver_filter)
# on partitionne par driver et on ordonne par lap number decroissant pour avoir le dernier tour de chaque driver
w = Window.partitionBy('Driver').orderBy(F.col('LapNumber').desc())
one_row_per_driver = (
    df.filter(F.col('Driver').isin(driver_filter))
      .withColumn('rn', F.row_number().over(w))
      .filter(F.col('rn') == 1) # permet de prendre la dernière ligne de chaque driver (dernier tour)
      .drop('rn')
)
one_row_per_driver.show(len(driver_filter), truncate=False)


# SHOW INFO

# Affiche les meilleurs tours par pilote (meilleur = temps minimal).
# On récupère tous les pilotes triés par meilleur tour
# on convertit en pandas et on affiche tout dans le terminal. (meilleur affichage)
print('-----------------------\nMeilleurs tours par pilote (tous les pilotes, triés par BestLap) :')
best_laps = df.groupBy('Driver')\
    .agg(F.min('LapTime_s').alias('BestLap'))\
    .orderBy('BestLap')
    
# pandas pour lisibilité 
best_laps_pdf = best_laps.toPandas()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', None)
best_laps_pdf['BestLap'] = best_laps_pdf['BestLap'].round(3)
print(best_laps_pdf.to_string(index=False))

# normalisation de la progression de la course (pour les courses où c'est en pourcentage, on divise par 100)
df = df.withColumn('RaceProgress_norm',
                   F.when(F.col('RaceProgress') > 1, F.col('RaceProgress')/100.0)
                    .otherwise(F.col('RaceProgress')))
