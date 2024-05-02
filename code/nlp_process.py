
import os
import sys
import logging
import argparse
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", nltk])


# Import pyspark and build Spark session
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from sparknlp.annotator import Normalizer
from sparknlp.annotator import LemmatizerModel
from nltk.corpus import stopwords
from pyspark.ml.feature import CountVectorizer , IDF
from pyspark.ml.clustering import LDA
from pyspark.sql.types import StringType, ArrayType, FloatType
from itertools import chain
import json
import sparknlp
import numpy as np
import pandas as pd
from sparknlp.base import *
from pyspark.ml import Pipeline
from sparknlp.annotator import *
import pyspark.sql.functions as f
from pyspark.sql import SparkSession

logging.basicConfig(format='%(asctime)s,%(levelname)s,%(module)s,%(filename)s,%(lineno)d,%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.info(f'defining function for later')
def indices_to_terms(indices, terms=terms):
        terms_subset = [terms[index] for index in indices]
        return terms_subset

def main():
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    parser.add_argument("--s3_dataset_path", type=str, help="Path of dataset in S3")
    parser.add_argument("--s3_output_bucket", type=str, help="s3 output bucket")
    parser.add_argument("--s3_output_key_prefix", type=str, help="s3 output key prefix")
    args = parser.parse_args()
    logger.info(f"args={args}")
    
    logger.info(f'spark session')
    spark = SparkSession.builder.appName("PySparkApp").getOrCreate()
    
    logger.info(f"Spark version: {spark.version}")
    logger.info(f"sparknlp version: {sparknlp.version()}")
    
    # This is needed to save RDDs which is the only way to write nested Dataframes into CSV format
    sc = spark.sparkContext
    sc._jsc.hadoopConfiguration().set(
        "mapred.output.committer.class", "org.apache.hadoop.mapred.FileOutputCommitter"
    )

    # Downloading the data from S3 into a Dataframe
    #Your code here
    print(f"going to read {args.s3_dataset_path}")
    output_prefix_data_comments = "project/comments/yyyy=*"
    s3_path = f"s3a://{bucket}/{output_prefix_data_comments}"
    comments = spark.read.parquet(s3_path, header=True).select('title', 'id')
    print(f"finished reading files...")
    
    logger.info(f"finished reading file")
    
    logger.info(f'creating small df')
    small_df = posts.select('title', 'id')
    
    loger.info(f'creating tokenizer')
    tokenizer = Tokenizer(outputCol="words")
    tokenizer.setInputCol("title")
    
    logger.info(f'creating normalizer')
    normalizer = Normalizer() \
     .setInputCols(['words']) \
     .setOutputCol('normalized') \
     .setCleanupPatterns(["""[^\w\d\s]"""]) \
     .setLowercase(True)
    
    logger.info(f'creating lemmatizer')
    lemmatizer = LemmatizerModel.pretrained() \
     .setInputCols(['normalized']) \
     .setOutputCol('lemmatized')
    
    loger.info(f'removing stopwords')
    stop_words = list(get_stop_words('en')) 
    #StopWords = stopwords.words("english")
    remover = StopWordsRemover(stopWords=stop_words)
    remover.setInputCol("lemmatized")
    remover.setOutputCol("filtered")
    
    logger.info(f'creating count vectorizer')
    cv = CountVectorizer(inputCol="filtered", outputCol="raw_features", vocabSize=5000, minDF=25)
    
    logger.info(f'creating idf')
    idf = IDF(inputCol="raw_features", outputCol="features")
    
    logger.info(f'creating lda')
    lda = LDA()
    lda.setK(8)
    lda.setMaxIter(10)
    lda.setSeed(13)
    
    logger.info(f'creating pipeline')
    pipeline = Pipeline(stages=[tokenizer, normalizer, lemmatizer, remover, cv, idf, lda])
    
    logger.info(f'fitting pipeline')
    model = pipeline.fit(small_df)
    
    logger.info('getting topics top words')
    udf_indices_to_terms = f.udf(indices_to_terms, ArrayType(StringType()))
    
    logger.info(f'topics df')
    topics = (
    topics
       .withColumn("terms", udf_indices_to_terms(f.col("termIndices")))
    )
    
    logger.info(f'showing topics')
    logger.info(topics.take(8))
    logger.info(f'trying to save topic info')
    
    s3_path = "s3://" + os.path.join('us-east-1:562166416351', "project-data","topics_info")
    logger.info(f"going to save dataframe to {s3_path}")
    topics.coalesce(1).write.format('csv').option('header', 'false').mode("overwrite").save(s3_path)
    
    logger.info("all done")
    
if __name__ == "__main__":
    main()
