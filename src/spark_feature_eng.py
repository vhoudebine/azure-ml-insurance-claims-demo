import argparse
from operator import add
from pyspark.sql.functions import col, sum, lag
from pyspark.sql.window import Window
import os 


parser = argparse.ArgumentParser()
parser.add_argument("--raw_data")
parser.add_argument("--training_data")

args = parser.parse_args()
print(args.raw_data)
print(args.training_data)

df = spark.read.parquet(args.raw_data)
days = lambda i: i * 86400 

window_spec = Window.partitionBy("fuel_type").orderBy(col("policy_created_date").cast("long")).rangeBetween(-days(30), 0)
df = df.withColumn("30_d_fuel_type_is_claim_sum", sum(col("is_claim")).over(window_spec))

print (f"Writing to {args.training_data}")

df.write.mode('overwrite').parquet(training_data_path)

