import os
import sys
import yaml
import logging

from functools import reduce

from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.context import SparkContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType, TimestampType, LongType
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def days_between(end, start):
    return (F.col(end).cast("long") - F.col(start).cast("long")) / 86400      

def get_inpatient(visit_occurrence, inpatient_visit_concept_id, emergency_visit_concept_id, admission_gap, er_lead_in_threshold):
    w = Window.partitionBy("person_id").orderBy("visit_start_datetime", "visit_end_datetime")
    w_prior = w.rowsBetween(Window.unboundedPreceding, -1)
    w_current = w.rowsBetween(Window.unboundedPreceding, Window.currentRow)

    # drop patients whose last visit occurs within 30 days of lastest datetime in visit_occurrence dataset
    latest_row = visit_occurrence.agg(F.max(F.col("visit_end_datetime")).alias("latest_datetime")).first()
    latest_datetime = latest_row["latest_datetime"]
    inpatient = (
        visit_occurrence
        .dropDuplicates(["person_id", "visit_occurrence_id", "visit_start_datetime", "visit_end_datetime"])
        .filter(F.col("visit_concept_id") == inpatient_visit_concept_id)
        .select("person_id", "visit_occurrence_id", "visit_concept_id", "visit_start_datetime", "visit_end_datetime")
        .orderBy(["person_id", "visit_start_datetime", "visit_end_datetime"])
        .withColumn("max_prior_visit_end_datetime", F.max("visit_end_datetime").over(w_prior))
        .withColumn("gap_days", days_between("visit_start_datetime", "max_prior_visit_end_datetime"))
        .withColumn("is_new_admission", F.when((F.col("gap_days") > 0) | (F.col("gap_days").isNull()), F.lit(1)).otherwise(F.lit(0)))
        .withColumn("admission_group", F.sum(F.col("is_new_admission")).over(w_current))
    )

    inpatient_agg = (
        inpatient
        .groupBy(["person_id", "admission_group"])
        .agg(F.min(F.col("visit_start_datetime")).alias("inpatient_visit_start_datetime"), F.max(F.col("visit_end_datetime")).alias("inpatient_visit_end_datetime"))
    )

    er = (
        visit_occurrence
        .dropDuplicates(["person_id", "visit_occurrence_id", "visit_start_datetime", "visit_end_datetime"])
        .filter(F.col("visit_concept_id") == emergency_visit_concept_id)
        .select("person_id", "visit_start_datetime", "visit_end_datetime")
        .orderBy(["person_id", "visit_start_datetime", "visit_end_datetime"])
    )


    w_admission =  Window.partitionBy("person_id").orderBy("inpatient_visit_earliest_start_datetime", "inpatient_visit_end_datetime")
    w_admission_prior = w_admission.rowsBetween(Window.unboundedPreceding, -1)

    er_lead_in = (
        inpatient_agg
        .join(er, "person_id", "inner")
        .withColumn("diff_start_times_days", days_between("inpatient_visit_start_datetime", "visit_start_datetime"))
        .withColumn("er_to_inpatient_gap_days", days_between("inpatient_visit_start_datetime", "visit_end_datetime"))
        .withColumn("er_lead_in", (F.col("diff_start_times_days") > 0) & (F.col("er_to_inpatient_gap_days") >= 0) & (F.col("er_to_inpatient_gap_days") <= er_lead_in_threshold))
        .filter(F.col("er_lead_in"))
        .groupBy(["person_id", "admission_group"])
        .agg(F.min("visit_start_datetime").alias("er_start_datetime"))
    )
    er_to_inpatient = (
        inpatient_agg.join(er_lead_in, ["person_id", "admission_group"], "left")
        .withColumn("earliest_start_datetime", F.least(F.col("inpatient_visit_start_datetime"), F.col("er_start_datetime")))
        .withColumn("inpatient_visit_earliest_start_datetime", F.when(F.col("earliest_start_datetime").isNotNull(), F.col("earliest_start_datetime")).otherwise(F.col("inpatient_visit_start_datetime")))
        .withColumn("length_of_admission_days", F.floor(days_between("inpatient_visit_end_datetime", "inpatient_visit_earliest_start_datetime")))
        .withColumn("prior_admission_end_datetime", F.lag("inpatient_visit_end_datetime").over(w_admission))
        .withColumn("next_admission_start_datetime", F.lead("inpatient_visit_earliest_start_datetime").over(w_admission))
        .withColumn("length_of_prior_gap_days", days_between("inpatient_visit_earliest_start_datetime", "prior_admission_end_datetime"))
        .withColumn("length_of_next_gap_days", days_between("next_admission_start_datetime", "inpatient_visit_end_datetime"))
        .withColumn("prior_admission_within_30_days", F.col("length_of_prior_gap_days").between(0, admission_gap).cast("int"))
        .withColumn("next_admission_within_30_days", F.col("length_of_next_gap_days").between(0, admission_gap).cast("int"))
        .withColumn("length_of_prior_admission_days", F.lag("length_of_admission_days").over(w_admission))
        .withColumn("number_of_prior_admissions", F.count("*").over(w_admission_prior))
        .filter((F.lit(latest_datetime).cast("long") - F.col("inpatient_visit_end_datetime").cast("long")) / 86400.0 > admission_gap) # censor these records only at the end so that records earlier than these can still have readmission within 30 days 
        .select(["person_id", "admission_group", "inpatient_visit_earliest_start_datetime", "inpatient_visit_end_datetime", "length_of_admission_days", "prior_admission_within_30_days", 
                 "length_of_prior_gap_days", "length_of_prior_admission_days", "number_of_prior_admissions", "next_admission_within_30_days"])
    )
    return er_to_inpatient

def get_person(person, inpatient):
    person_cleaned = person.dropDuplicates(["person_id"])
    person_merged = (
        inpatient
            .join(person_cleaned, "person_id", "inner")
            .withColumn("age_at_inpatient_visit_years", F.floor(F.months_between(F.col("inpatient_visit_earliest_start_datetime"), F.col("birth_datetime")) / 12))
            .select(["person_id", "inpatient_visit_earliest_start_datetime", "inpatient_visit_end_datetime", "age_at_inpatient_visit_years", "gender_concept_id", "race_concept_id", "ethnicity_concept_id"])
    )
    return person_merged

def get_condition(condition_occurrence, inpatient):
    condition_occurrence = condition_occurrence.withColumn("new_condition_concept_id", F.coalesce(F.when(F.col("condition_concept_id")==0, None).otherwise(F.col("condition_concept_id")), F.col("condition_source_concept_id")))
    condition_merged = (
        inpatient
        .join(condition_occurrence, "person_id", "inner")
        .filter(F.col("inpatient_visit_end_datetime") >= F.col("condition_start_datetime"))
    )
    inpatient_condition = (
        condition_merged
        .groupBy(["person_id", "inpatient_visit_earliest_start_datetime", "inpatient_visit_end_datetime"])
        .agg(F.countDistinct(F.col("new_condition_concept_id")).alias("number_of_distinct_conditions"), F.array_sort(F.collect_set(F.col("new_condition_concept_id").cast("string"))).alias("distinct_conditions_set"))
        .orderBy("person_id", "inpatient_visit_earliest_start_datetime", "inpatient_visit_end_datetime")
    )
    return inpatient_condition

def get_drug(drug_exposure, inpatient):
    drug_exposure = drug_exposure.withColumn("new_drug_concept_id", F.coalesce(F.when(F.col("drug_concept_id")==0, None).otherwise(F.col("drug_concept_id")), F.col("drug_source_concept_id")))
    drug_merged = (
        inpatient
        .join(drug_exposure, "person_id", "inner") 
        .filter(F.col("inpatient_visit_end_datetime") >= F.col("drug_exposure_start_datetime"))
    )
    inpatient_drug = (
        drug_merged.groupBy(["person_id", "inpatient_visit_earliest_start_datetime", "inpatient_visit_end_datetime"])
        .agg(F.countDistinct("new_drug_concept_id").alias("number_of_distinct_drugs"), F.array_sort(F.collect_set(F.col("new_drug_concept_id").cast("string"))).alias("distinct_drugs_set"))
        .orderBy("person_id", "inpatient_visit_earliest_start_datetime", "inpatient_visit_end_datetime")
    )
    return inpatient_drug

def get_procedure(procedure_occurrence, inpatient):
    procedure_occurrence = procedure_occurrence.withColumn("new_procedure_concept_id", F.coalesce(F.when(F.col("procedure_concept_id")==0, None).otherwise(F.col("procedure_concept_id")), F.col("procedure_source_concept_id")))
    procedure_merged = (
        inpatient
        .join(procedure_occurrence, "person_id", "inner")
        .filter(F.col("inpatient_visit_end_datetime") >= F.col("procedure_datetime"))
    )
    inpatient_procedure = (
        procedure_merged
        .groupBy(["person_id", "inpatient_visit_earliest_start_datetime", "inpatient_visit_end_datetime"])
        .agg(F.countDistinct(F.col("new_procedure_concept_id")).alias("number_of_distinct_procedures"), F.array_sort(F.collect_set(F.col("new_procedure_concept_id").cast("string"))).alias("distinct_procedures_set"))
        .orderBy("person_id", "inpatient_visit_earliest_start_datetime", "inpatient_visit_end_datetime")
    )
    return inpatient_procedure

def get_feature_table(df, fillna_cols, vec_cols):
    df = df.fillna(0, subset=[c for c in fillna_cols])
    for c in vec_cols:
        df = df.withColumn(c, F.coalesce(F.col(c), F.array().cast(f"array<{'string'}>")))
    # any_null = reduce(lambda a, b: a | b, [F.col(c).isNull() for c in df.columns])
    # df.filter(any_null).show()
    return df

def combine_dataframes(visit_occurrence, person, condition_occurrence, drug_exposure, procedure_occurrence, inpatient_visit_concept_id, emergency_visit_concept_id, admission_gap, er_lead_in_threshold):
    inpatient= get_inpatient(visit_occurrence, inpatient_visit_concept_id, emergency_visit_concept_id, admission_gap, er_lead_in_threshold)
    inpatient_person = get_person(person, inpatient)
    inpatient_condition = get_condition(condition_occurrence, inpatient)
    inpatient_drug = get_drug(drug_exposure, inpatient)
    inpatient_procedure = get_procedure(procedure_occurrence, inpatient)

    df = (
        inpatient
        .join(inpatient_person, ["person_id", "inpatient_visit_earliest_start_datetime", "inpatient_visit_end_datetime"], "left")
        .join(inpatient_condition, ["person_id", "inpatient_visit_earliest_start_datetime", "inpatient_visit_end_datetime"], "left")
        .join(inpatient_drug, ["person_id", "inpatient_visit_earliest_start_datetime", "inpatient_visit_end_datetime"], "left")
        .join(inpatient_procedure, ["person_id", "inpatient_visit_earliest_start_datetime", "inpatient_visit_end_datetime"], "left")
        .orderBy("person_id", "inpatient_visit_earliest_start_datetime", "inpatient_visit_end_datetime")
        .dropDuplicates()
        )
    return df


def main():
    args = getResolvedOptions(sys.argv, ["CONFIG_PATH", "JOB_NAME", "INPUT_PREFIX", "OUTPUT_PREFIX"])

    config_filename = os.path.basename(args["CONFIG_PATH"])
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)
    
    sc = SparkContext.getOrCreate()
    glue_context = GlueContext(sc)
    spark = glue_context.spark_session

    JOB_NAME = args["JOB_NAME"]

    job = Job(glue_context)
    job.init(JOB_NAME, args)

    print(f"Starting job {JOB_NAME}...")

    INPUT_PREFIX = args["INPUT_PREFIX"]
    OUTPUT_PREFIX = args["OUTPUT_PREFIX"]
    
    INPATIENT_VISIT_CONCEPT_ID = config["concept_ids"]["inpatient_visit_concept_id"]
    EMERGENCY_VISIT_CONCEPT_ID = config["concept_ids"]["emergency_visit_concept_id"]

    ADMISSION_GAP = config["admission_gap"]
    ER_LEAD_IN_THRESHOLD = config["er_lead_in_threshold"]

    vec_cols = ["distinct_conditions_set", "distinct_drugs_set", "distinct_procedures_set"]
    fillna_cols = ["prior_admission_within_30_days", "next_admission_within_30_days", "number_of_distinct_conditions", "number_of_distinct_drugs", "number_of_distinct_procedures"]

    visit_schema = StructType([
        StructField("visit_occurrence_id", IntegerType(), True),
        StructField("person_id", LongType(), True),
        StructField("visit_concept_id", IntegerType(), True),
        StructField("visit_start_date", DateType(), True),
        StructField("visit_start_datetime", TimestampType(), True),
        StructField("visit_end_date", DateType(), True),
        StructField("visit_end_datetime", TimestampType(), True),
        StructField("visit_type_concept_id", IntegerType(), True),
        StructField("provider_id", IntegerType(), True),
        StructField("care_site_id", IntegerType(), True),
        StructField("visit_source_value", StringType(), True),
        StructField("visit_source_concept_id", IntegerType(), True),
        StructField("admitting_source_concept_id", IntegerType(), True),
        StructField("admitting_source_value", StringType(), True),
        StructField("discharge_to_concept_id", IntegerType(), True),
        StructField("discharge_to_source_value", StringType(), True),
        StructField("preceding_visit_occurrence_id", IntegerType(), True),
    ])

    person_schema = StructType([
        StructField("person_id", LongType(), True),
        StructField("gender_concept_id", IntegerType(), True),
        StructField("year_of_birth", IntegerType(), True),
        StructField("month_of_birth", IntegerType(), True),
        StructField("day_of_birth", IntegerType(), True),
        StructField("birth_datetime", TimestampType(), True),
        StructField("race_concept_id", IntegerType(), True),
        StructField("ethnicity_concept_id", IntegerType(), True),
        StructField("location_id", IntegerType(), True),
        StructField("provider_id", IntegerType(), True),
        StructField("care_site_id", IntegerType(), True),
        StructField("person_source_value", StringType(), True),
        StructField("gender_source_value", StringType(), True),
        StructField("gender_source_concept_id", IntegerType(), True),
        StructField("race_source_value", StringType(), True),
        StructField("race_source_concept_id", IntegerType(), True),
        StructField("ethnicity_source_value", StringType(), True),
        StructField("ethnicity_source_concept_id", IntegerType(), True),
    ])

    condition_schema = StructType([
        StructField("condition_occurrence_id", IntegerType(), True),
        StructField("person_id", LongType(), True),
        StructField("condition_concept_id", IntegerType(), True),
        StructField("condition_start_date", DateType(), True),
        StructField("condition_start_datetime", TimestampType(), True),
        StructField("condition_end_date", DateType(), True),
        StructField("condition_end_datetime", TimestampType(), True),
        StructField("condition_type_concept_id", IntegerType(), True),
        StructField("stop_reason", StringType(), True),
        StructField("provider_id", IntegerType(), True),
        StructField("visit_occurrence_id", IntegerType(), True),
        StructField("visit_detail_id", IntegerType(), True),
        StructField("condition_source_value", StringType(), True),
        StructField("condition_source_concept_id", IntegerType(), True),
        StructField("condition_status_source_value", StringType(), True),
        StructField("condition_status_concept_id", IntegerType(), True),
    ])

    drug_schema = StructType([
        StructField("drug_exposure_id", IntegerType(), True),
        StructField("person_id", LongType(), True),
        StructField("drug_concept_id", IntegerType(), True),
        StructField("drug_exposure_start_date", DateType(), True),
        StructField("drug_exposure_start_datetime", TimestampType(), True),
        StructField("drug_exposure_end_date", DateType(), True),
        StructField("drug_exposure_end_datetime", TimestampType(), True),
        StructField("verbatim_end_date", DateType(), True),
        StructField("drug_type_concept_id", IntegerType(), True),
        StructField("stop_reason", StringType(), True),
        StructField("refills", StringType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("days_supply", IntegerType(), True),
        StructField("sig", StringType(), True),
        StructField("route_concept_id", IntegerType(), True),
        StructField("lot_number", IntegerType(), True),
        StructField("provider_id", IntegerType(), True),
        StructField("visit_occurrence_id", IntegerType(), True),
        StructField("visit_detail_id", IntegerType(), True),
        StructField("drug_source_value", StringType(), True),
        StructField("drug_source_concept_id", IntegerType(), True),
        StructField("route_source_value", StringType(), True),
        StructField("dose_unit_source_value", StringType(), True),
    ])

    procedure_schema = StructType([
        StructField("procedure_occurrence_id", IntegerType(), True),
        StructField("person_id", LongType(), True),
        StructField("procedure_concept_id", IntegerType(), True),
        StructField("procedure_date", DateType(), True),
        StructField("procedure_datetime", TimestampType(), True), 
        StructField("procedure_type_concept_id", IntegerType(), True),
        StructField("modifier_concept_id", IntegerType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("provider_id", IntegerType(), True),
        StructField("visit_occurrence_id", IntegerType(), True),
        StructField("visit_detail_id", IntegerType(), True),
        StructField("procedure_source_value", StringType(), True),
        StructField("procedure_source_concept_id", IntegerType(), True),
        StructField("modifier_source_value", StringType(), True),
    ])

    visit_occurrence = spark.read.parquet(f"{INPUT_PREFIX}/visit_occurrence/", header=True)
    person = spark.read.parquet(f"{INPUT_PREFIX}/person/", header=True)
    condition_occurrence = spark.read.parquet(f"{INPUT_PREFIX}/condition_occurrence/", header=True)
    drug_exposure = spark.read.parquet(f"{INPUT_PREFIX}/drug_exposure/", header=True)
    procedure_occurrence = spark.read.parquet(f"{INPUT_PREFIX}/procedure_occurrence/", header=True)
    df = combine_dataframes(visit_occurrence, person, condition_occurrence, drug_exposure, procedure_occurrence, 
                            INPATIENT_VISIT_CONCEPT_ID, EMERGENCY_VISIT_CONCEPT_ID, ADMISSION_GAP, ER_LEAD_IN_THRESHOLD)
    feature_table = get_feature_table(df, fillna_cols, vec_cols)
    feature_table.coalesce(1).write.mode("overwrite").parquet(OUTPUT_PREFIX)
    print(f"Feature table created at {OUTPUT_PREFIX}!")
    job.commit()


if __name__ == "__main__":
    main()
