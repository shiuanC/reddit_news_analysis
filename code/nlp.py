

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
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from sparknlp.pretrained import PretrainedPipeline

logging.basicConfig(format='%(asctime)s,%(levelname)s,%(module)s,%(filename)s,%(lineno)d,%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

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
    
    import sagemaker
    session = sagemaker.Session()
    bucket = session.default_bucket()
    output_prefix_data_submissions = "project/submissions/yyyy=*"
    s3_path = f"s3a://{bucket}/{output_prefix_data_submissions}"
    print(f"reading submissions from {s3_path}")
    posts = spark.read.parquet(s3_path)
    
    #reading comments
    output_prefix_data_comments = "project/comments/yyyy=*"
    s3_path = f"s3a://{bucket}/{output_prefix_data_comments}"
    comments = spark.read.parquet(s3_path)
    
    
    
    import pyspark.sql.functions as f
    comments = comments.withColumn('misinfo_class', 
                    f.when(comments.body.rlike(r'fake news|bullshit|misinfo|clickbait|unreliable|propoganda'), True)\
                    .otherwise(False))
    
    #from pyspark.ml.feature import Tokenizer, StopWordsRemover
    from sparknlp.base import DocumentAssembler, Finisher
    from sparknlp.annotator import Lemmatizer, Normalizer, Tokenizer, StopWordsCleaner
    from pyspark.ml import Pipeline
    from nltk.corpus import stopwords
    from pyspark.ml.feature import CountVectorizer , IDF
    from pyspark.ml.clustering import LDA
    from pyspark.sql.types import StringType, ArrayType, FloatType
    from itertools import chain
    
    small_df = posts.select('title', 'id')
    
    documentAssembler = DocumentAssembler()       
    documentAssembler.setInputCol('title')      
    documentAssembler.setOutputCol('document')
    
    tokenizer = Tokenizer() 
    tokenizer.setInputCols(["document"]) 
    tokenizer.setOutputCol("token")
    
    normalizer = Normalizer() \
     .setInputCols(['token']) \
     .setOutputCol('normalized') \
     .setLowercase(True)
    
    #remove stop words
    StopWords = stopwords.words("english")
    #adding news sources and stopwords in other languages
    StopWords += ['reuters', 'في','از', 'ap', 'says', 'bbcworld', 'amp', 'rt', 'apentertainment', 'la', 'user', 'deleted',
             'क', 'di','در','آهنگ', 'de', 'el', 'en','دانلود', 'به', 'म']
    stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setStopWords(StopWords)\
      .setCaseSensitive(False)
    
    finisher = Finisher() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCols(["tokens"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)
    
    #count vectorizer
    cv = CountVectorizer(inputCol="tokens", outputCol="raw_features", vocabSize=5000, minDF=25)
    # IDF
    idf = IDF(inputCol="raw_features", outputCol="features")
    
    #lda model 
    lda = LDA()
    lda.setK(8)
    lda.setMaxIter(20)
    lda.setSeed(13)
    
    pipeline = Pipeline(stages=[documentAssembler, 
                            tokenizer, 
                            normalizer, 
                            stopwords_cleaner, 
                            finisher, 
                            cv, 
                            idf, 
                            lda])
    
    model = pipeline.fit(small_df)
    
    small_df = model.transform(small_df)
    
    topics = model.stages[-1].describeTopics()
    terms = model.stages[-3].vocabulary
    
    #get word from index of term 
    def indices_to_terms(indices, terms=terms):
        terms_subset = [terms[index] for index in indices]
        return terms_subset
# Defining Spark UDF from above function
    udf_indices_to_terms = f.udf(indices_to_terms, ArrayType(StringType()))

    topics = (
        topics
           .withColumn("terms", udf_indices_to_terms(f.col("termIndices")))
        )
    
    topic_dict = {0: 'russia&ukraine', 1: 'social media', 2: 'current events', 3: 'tv shows', 4: 'covid', 
              5: 'foriegn relations', 6: 'emerging tech', 7: 'demographic info'}
    
    mapping_expr = f.create_map([f.lit(x) for x in chain(*topic_dict.items())])
    
    
    #udf to get the top topic 
    max_topic = f.udf(lambda v:float(v.argmax()),FloatType())
    #using mao and udf to create a topic column
    topic = small_df.withColumn('topic_num', max_topic("topicDistribution"))\
        .withColumn("topic", mapping_expr[f.col("topic_num")]).select('id','topic')
    
    mini_posts = posts.select('created_utc', 'title', 'id')
    
    merged_df = mini_posts.join(topic, 'id')
    
    mini_comments = comments\
    .withColumn('comment_created', f.col('created_utc')).withColumn('comment_id', f.col('id'))\
    .withColumn('id', f.regexp_extract('link_id', 't3_(.*)$', 1))\
    .select('comment_created','body','misinfo_class', 'link_id', 'id')
    
    total_df = merged_df.join(mini_comments, 'id')
    
    s3_path = f"s3://{args.s3_output_bucket}/{args.s3_output_prefix}/comments"
    
    total_df.write.mode("overwrite").parquet(s3_path)
    
if __name__ == "__main__":
    main()
