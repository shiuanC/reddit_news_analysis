
import os
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
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

logging.basicConfig(format='%(asctime)s,%(levelname)s,%(module)s,%(filename)s,%(lineno)d,%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def main():
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    parser.add_argument("--s3_dataset_path", type=str, help="Path of dataset in S3")    
    parser.add_argument("--s3_output_bucket", type=str, help="s3 output bucket")
    parser.add_argument("--s3_output_prefix", type=str, help="s3 output prefix")
    parser.add_argument("--col_name_for_filtering", type=str, help="Name of the column to filter")
    parser.add_argument("--values_to_keep", type=str, help="comma separated list of values to keep in the filtered set")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("PySparkApp").getOrCreate()
    logger.info(f"spark version = {spark.version}")
    
    # This is needed to save RDDs which is the only way to write nested Dataframes into CSV format
    sc = spark.sparkContext
    sc._jsc.hadoopConfiguration().set(
        "mapred.output.committer.class", "org.apache.hadoop.mapred.FileOutputCommitter"
    )

   
    # Downloading the data from S3 into a Dataframe
    logger.info(f"going to read {args.s3_dataset_path}")
    df = spark.read.parquet(args.s3_dataset_path, header=True)
    logger.info(f"finished reading files...")
    

    
    # filter the dataframe to only keep the values of interest
    vals = [s.strip() for s in args.values_to_keep.split(",")]
    df_filtered = df.where(col(args.col_name_for_filtering).isin(vals))
    
    # save the filtered dataframes so that these files can now be used for future analysis
    s3_path = f"s3://{args.s3_output_bucket}/{args.s3_output_prefix}"
    logger.info(f"going to write data for {vals} in {s3_path}")
    logger.info(f"shape of the df_filtered dataframe is {df_filtered.count():,}x{len(df_filtered.columns)}")
    df_filtered.write.mode("overwrite").parquet(s3_path)
    
    logger.info(f"all done...")
    
if __name__ == "__main__":
    main()
