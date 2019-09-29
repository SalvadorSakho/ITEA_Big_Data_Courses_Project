
import random
import shutil
from pathlib import Path

import pandas as pd
from pyspark import SparkContext, SQLContext, SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.sql.functions import lit
from pyspark.sql.types import DoubleType

HOME_DIR = Path('.').resolve()
CSV_DATA_SOURCE = HOME_DIR / 'datasource' / 'csv'
JSON_DATA_SOURCE = HOME_DIR / 'datasource' / 'json'
ZIP_DATA_SOURCE = HOME_DIR / 'datasource' / 'zip'

DATA_PARQUET = HOME_DIR / 'output' / 'parquet'
DATA_CSV = HOME_DIR / 'output' / 'csv'

conf = SparkConf().setMaster("local").setAppName("YoutubeInvestigator").set("spark.network.timeout", "1000s")
spark = SparkContext(conf=conf)
sql = SQLContext(spark)


def process_data():
    statistic_data = sql.read.csv(str(CSV_DATA_SOURCE))
    statistic_data.registerTempTable('youtube_trends')
    statistic_data.cache()

    description_data = sql.read.json(
        str(JSON_DATA_SOURCE), multiLine=True
    )
    description_data.registerTempTable('youtube_trends_explanation')
    description_data.cache()

    description_data_frame = sql.sql(
        'select items.id, '
        'items.snippet.title '
        'from youtube_trends_explanation '
    ).toDF('id', 'title')

    statistic_data_frame = sql.sql(
        'select '
        '_c0 as country_abbreviation, '
        'cast(count(*) as int) as total_quantity_of_post_daily, '
        'cast(_c5 as int) as category_id '
        'from youtube_trends '
        'where _c5 <> "KSI" and _c5 is not null '
        'group by _c0, _c5 '
        'order by _c0'
    )

    if Path(f'{DATA_PARQUET}\\general_statistic_data.parquet').exists():
        shutil.rmtree(f'{DATA_PARQUET}\\general_statistic_data.parquet', ignore_errors=True)
    statistic_data_frame.write.partitionBy('country_abbreviation') \
        .parquet(f'{DATA_PARQUET}\\general_statistic_data.parquet')

    for value in description_data_frame.collect():
        df = pd.DataFrame(value.asDict())
    final_df = sql.createDataFrame(df)

    if Path(f'{DATA_PARQUET}\\data_for_processing_data.parquet').exists():
        shutil.rmtree(f'{DATA_PARQUET}\\data_for_processing_data.parquet', ignore_errors=True)
    statistic_data_frame.join(final_df, statistic_data_frame.category_id == final_df.id).withColumn('test_flag' \
        , lit(random.randint(0, 1))).write.parquet(f'{DATA_PARQUET}\\data_for_processing_data.parquet')

    statistic_data_frame.join(final_df, statistic_data_frame.category_id == final_df.id) \
         .write.csv(f'{DATA_CSV}\\data_for_processing_data.csv')


def machine_learning():
    df = sql.read.parquet(str(f'{DATA_PARQUET}\\data_for_processing_data.parquet'))
    df = df.withColumn('label', df.total_quantity_of_post_daily.cast(DoubleType()))
    df.show()
    features = ['total_quantity_of_post_daily', 'category_id']
    variables = features + ['test_flag']
    pipeline_prepare = Pipeline(stages=[
        VectorAssembler(inputCols=features, outputCol='features'),
    ])

    prepared = pipeline_prepare.fit(df).transform(df.dropna(subset=variables))
    training = prepared.filter(col('test_flag') == 0)
    testing = prepared.filter(col('test_flag') == 1)
    training_small = training.sample(fraction=0.3, seed=100500)

    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='target')

    classifier = RandomForestClassifier(featuresCol='features', labelCol='label')
    model = classifier.fit(training_small)
    predicted = model.transform(testing)
    print('Test Area Under ROC: ', evaluator.evaluate(predicted))


def main():
    process_data()
    machine_learning()


if __name__ == '__main__':
    main()
