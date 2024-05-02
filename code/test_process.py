
import os
import sys
import logging
import argparse

# Import pyspark and build Spark session
from pyspark.sql.functions import *
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

import json
import sparknlp
import numpy as np
import pandas as pd
from sparknlp.base import *
from pyspark.ml import Pipeline
from sparknlp.annotator import *
import pyspark.sql.functions as f
from pyspark.sql import SparkSession
from sparknlp.pretrained import PretrainedPipeline

logging.basicConfig(format='%(asctime)s,%(levelname)s,%(module)s,%(filename)s,%(lineno)d,%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def clean_text(df):
    # Lowercase all text
    df = df.withColumn("body", f.lower(f.col("body")))
    # Remove special characters (keeping only alphanumeric and spaces)
    df = df.withColumn("body", f.regexp_replace(f.col("body"), "[^a-zA-Z0-9\\s]", ""))
    # Trim spaces
    df = df.withColumn("body", f.trim(f.col("body")))
    return df

#get word from index of term 
def indices_to_terms(indices, terms):
    terms_subset = [terms[index] for index in indices]
    return terms_subset

def main():
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    parser.add_argument("--s3_dataset_path", type=str, help="Path of dataset in S3")
    parser.add_argument("--s3_output_bucket", type=str, help="s3 output bucket")
    parser.add_argument("--s3_output_key_prefix", type=str, help="s3 output key prefix")
    args = parser.parse_args()
    logger.info(f"args={args}")
    
    spark = SparkSession.builder \
            .appName("Spark NLP")\
            .config("spark.driver.memory","16G")\
            .config("spark.driver.maxResultSize", "0") \
            .config("spark.kryoserializer.buffer.max", "2000M")\
            .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.3")\
            .getOrCreate()
    
    logger.info(f"Spark version: {spark.version}")
    logger.info(f"sparknlp version: {sparknlp.version()}")
    
    # This is needed to save RDDs which is the only way to write nested Dataframes into CSV format
    sc = spark.sparkContext
    sc._jsc.hadoopConfiguration().set(
        "mapred.output.committer.class", "org.apache.hadoop.mapred.FileOutputCommitter"
    )

     # # load dataset
    submissions_path=f"{args.s3_dataset_path}/submissions/yyyy=*"
    logger.info(f"submissions_path={submissions_path}")
    posts = spark.read.parquet(submissions_path, header=True)
    logger.info(f"posts count={posts.count()}")
    posts.printSchema()
    
    comments_path=f"{args.s3_dataset_path}/comments/yyyy=*"
    logger.info(f"comments path={comments_path}")
    comments = spark.read.parquet(comments_path, header=True)
    logger.info(f"comments count={comments.count()}")
    #some preprocess for comments
    comments = clean_text(comments)
    comments = comments.withColumn('misinfo_class', 
                    f.when(comments.body.rlike(r'fake news|bullshit|misinfo|clickbait|unreliable|propoganda|propaganda|fraud|deceptive|fabricated|deep state|wake up|truth about'), True)\
                    .otherwise(False))
    logger.info(f"Finish data preprocess")
    
    
if __name__ == "__main__":
    main()
