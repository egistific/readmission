import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd

from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

from pyspark.ml.evaluation import BinaryClassificationEvaluator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_classification_metrics(df, label_col, pred_col):
    tp = df.filter((F.col(label_col) == 1) & (F.col(pred_col) == 1)).count()
    tn = df.filter((F.col(label_col) == 0) & (F.col(pred_col) == 0)).count()
    fp = df.filter((F.col(label_col) == 0) & (F.col(pred_col) == 1)).count()
    fn = df.filter((F.col(label_col) == 1) & (F.col(pred_col) == 0)).count()
    div = lambda a, b: a / (b + 1e-6)
    metrics_dict = {
        "accuracy": div(tp + tn, tp + tn + fp + fn), 
        "sensitivity (recall)": div(tp, tp + fn), 
        "specificity": div(tn, tn + fp), 
        "PPV (precision)": div(tp, tp + fp), 
        "NPV": div(tn, tn + fn)
        }
    metrics_dict["f1"] = div(2 * metrics_dict["PPV (precision)"] * metrics_dict["sensitivity (recall)"], 
                             metrics_dict["PPV (precision)"] + metrics_dict["sensitivity (recall)"])
    return metrics_dict

def get_confusion_matrix(df, label_col, pred_col):
    df_ct = (
        df
        .stat.crosstab(label_col, pred_col)
        .toDF("actual", "pred_no_admission", "pred_readmission")
        .withColumn("actual", F.col("actual").cast("int"))
        .orderBy("actual")
        .toPandas())
    return df_ct


def main():
    args = getResolvedOptions(sys.argv, ["CONFIG_PATH", "JOB_NAME", "INPUT_PREFIX", "OUTPUT_PREFIX"])

    config_filename = os.path.basename(args["CONFIG_PATH"])
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)
    
    sc = SparkContext.getOrCreate()
    glue_context = GlueContext(sc)
    spark = glue_context.spark_session

    job = Job(glue_context)
    job.init(args["JOB_NAME"], args)

    JOB_NAME = args["JOB_NAME"]
    INPUT_PREFIX = args["INPUT_PREFIX"]
    OUTPUT_PREFIX = args["OUTPUT_PREFIX"]

    label = "next_admission_within_30_days"

    print(f"Starting job {JOB_NAME}...")
    df_train = spark.read.parquet(f"{INPUT_PREFIX}/train")
    df_test = spark.read.parquet(f"{INPUT_PREFIX}/test")

    df_train = (
        df_train
        .withColumn("prob", vector_to_array("probability")[1])
        .withColumn("prediction",F.col("prediction").cast("int"))
        # .select("person_id", "admission_group", "inpatient_visit_earliest_start_datetime", "inpatient_visit_end_datetime", label, F.col("prediction").cast("int"), "prob", "rawPrediction")
    )
    df_test = (
        df_test
        .withColumn("prob", vector_to_array("probability")[1])
        .withColumn("prediction",F.col("prediction").cast("int"))
        # .select("person_id", "admission_group", "inpatient_visit_earliest_start_datetime", "inpatient_visit_end_datetime", label, F.col("prediction").cast("int"), "prob", "rawPrediction")
    )

    evaluator = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    train_auc = evaluator.evaluate(df_train)
    test_auc = evaluator.evaluate(df_test)
    train_metrics = get_classification_metrics(df_train, label, "prediction")
    test_metrics = get_classification_metrics(df_test, label, "prediction")
    train_metrics["auc"] = train_auc
    test_metrics["auc"] = test_auc
    df_train_metrics = pd.DataFrame([train_metrics])
    df_test_metrics = pd.DataFrame([test_metrics])
    df_train_metrics.to_csv(f"{OUTPUT_PREFIX}/train/metrics.csv", index=False)
    df_test_metrics.to_csv(f"{OUTPUT_PREFIX}/test/metrics.csv", index=False)

    cm_train = get_confusion_matrix(df_train, label, "prediction")
    cm_test = get_confusion_matrix(df_test, label, "prediction")
    cm_train.to_csv(f"{OUTPUT_PREFIX}/train/cm.csv", index=False)
    cm_test.to_csv(f"{OUTPUT_PREFIX}/test/cm.csv", index=False)

    job.commit()


if __name__ == "__main__":
    main()

