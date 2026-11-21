import os
import sys
import yaml
import logging

from functools import reduce

from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

INPUT_TABLES = ["visit_occurrence", "person", "condition_occurrence", "drug_exposure", "procedure_occurrence"]

def days_between(end, start):
    return f"((unix_timestamp({end}) - unix_timestamp({start})) / 86400)"

def register_tables(spark, input_prefix):
    for t in INPUT_TABLES:
        spark.read.parquet(f"{input_prefix}/{t}/").createOrReplaceTempView(t)


def inpatient_group(spark, inpatient_visit_concept_id, emergency_visit_concept_id, er_lead_in_threshold, admission_gap):

    w_order = "PARTITION BY person_id ORDER BY visit_start_datetime, visit_end_datetime"
    w_prior = f"{w_order} ROWS BETWEEN UNBOUNDED PRECEDING and 1 PRECEDING"
    w_current = f"{w_order} ROWS BETWEEN UNBOUNDED PRECEDING and CURRENT ROW"


    query_rn = f"""
    SELECT person_id,
        visit_concept_id,
        visit_occurrence_id,
        visit_start_datetime,
        visit_end_datetime,
        ROW_NUMBER() OVER(
            PARTITION BY person_id, visit_occurrence_id, visit_start_datetime, visit_end_datetime
            ORDER BY visit_concept_id
            ) AS row_num
    FROM visit_occurrence
    """
    spark.sql(query_rn).createOrReplaceTempView("visit_occurrence_rn")

    query_dedup = f"""
    SELECT person_id,
        visit_concept_id,
        visit_occurrence_id,
        visit_start_datetime,
        visit_end_datetime
    FROM visit_occurrence_rn
    WHERE row_num = 1
    """
    spark.sql(query_dedup).createOrReplaceTempView("visit_occurrence_dedup")

    query_prior = f"""
    SELECT person_id,
        visit_concept_id,
        visit_occurrence_id,
        visit_start_datetime,
        visit_end_datetime,
        MAX(visit_end_datetime) OVER ({w_prior}) AS max_prior_visit_end_datetime 
    FROM visit_occurrence_dedup
    WHERE visit_concept_id = {inpatient_visit_concept_id}
    """
    spark.sql(query_prior).createOrReplaceTempView("inpatient_visits")

    query_gap = f"""
    SELECT *,
        {days_between("visit_start_datetime", "max_prior_visit_end_datetime")} AS gap_days 
    FROM inpatient_visits
    """
    spark.sql(query_gap).createOrReplaceTempView("inpatient_visit_gap")


    query_admission_group = f"""
    SELECT *,
        SUM(CASE WHEN gap_days > 0 or gap_days IS NULL THEN 1 ELSE 0 END) OVER ({w_current}) AS admission_group
    FROM inpatient_visit_gap
    """
    spark.sql(query_admission_group).createOrReplaceTempView("inpatient")

    query_inpatient_agg = f"""
    SELECT person_id,
           admission_group,
           MIN(visit_start_datetime) AS inpatient_visit_start_datetime,
           MAX(visit_end_datetime) AS inpatient_visit_end_datetime
    FROM inpatient
    GROUP BY person_id, admission_group
    """
    spark.sql(query_inpatient_agg).createOrReplaceTempView("inpatient_agg")

    query_er = f"""
    SELECT person_id,
           visit_start_datetime,
           visit_end_datetime
    FROM visit_occurrence_dedup
    where visit_concept_id = {emergency_visit_concept_id}
    """
    spark.sql(query_er).createOrReplaceTempView("er")

    w_admission_order = "PARTITION BY person_id ORDER BY inpatient_visit_earliest_start_datetime, inpatient_visit_end_datetime"
    w_admission_prior = f"{w_admission_order} ROWS BETWEEN UNBOUNDED PRECEDING and 1 PRECEDING"

    query_er_diff = f"""
    SELECT a.person_id,
           a.admission_group, 
           a.inpatient_visit_start_datetime, 
           a.inpatient_visit_end_datetime,
           e.visit_start_datetime,
           e.visit_end_datetime,
        {days_between("a.inpatient_visit_start_datetime", "e.visit_start_datetime")} AS diff_start_time_days,
        {days_between("a.inpatient_visit_start_datetime", "e.visit_end_datetime")} AS er_to_inpatient_gap_days
    FROM inpatient_agg a
    INNER JOIN er e
        ON a.person_id = e.person_id
   """
    spark.sql(query_er_diff).createOrReplaceTempView("er_diff")

    query_er_flagged = f"""
    SELECT person_id,
           admission_group, 
           inpatient_visit_start_datetime, 
           inpatient_visit_end_datetime,
           visit_start_datetime,
           visit_end_datetime,
           diff_start_time_days,
           er_to_inpatient_gap_days,
           diff_start_time_days > 0 AND er_to_inpatient_gap_days <= {er_lead_in_threshold} AS er_lead_in
    FROM er_diff
    """
    spark.sql(query_er_flagged).createOrReplaceTempView("er_flagged")

    query_er_lead_in = f"""
    SELECT person_id,
           admission_group, 
           MIN(visit_start_datetime) AS er_start_datetime
    FROM er_flagged
    WHERE er_lead_in
    GROUP BY person_id, admission_group
    """
    spark.sql(query_er_lead_in).createOrReplaceTempView("er_lead_in")

    query_er_join = f"""
    SELECT a.person_id,
           a.admission_group,
           a.inpatient_visit_start_datetime,
           a.inpatient_visit_end_datetime,
           e.er_start_datetime,
           COALESCE(e.er_start_datetime, a.inpatient_visit_start_datetime) AS inpatient_visit_earliest_start_datetime
    FROM inpatient_agg a
    LEFT JOIN er_lead_in e
        ON a.person_id = e.person_id 
        AND a.admission_group = e.admission_group 
    """
    spark.sql(query_er_join).createOrReplaceTempView("er_join")

    query_er_inpatient1 = f"""
    SELECT person_id, 
           admission_group, 
           inpatient_visit_earliest_start_datetime,
           inpatient_visit_end_datetime,
           FLOOR({days_between("inpatient_visit_end_datetime", "inpatient_visit_earliest_start_datetime")}) AS length_of_admission_days,
           LAG(inpatient_visit_end_datetime) OVER ({w_admission_order}) AS prior_admission_end_datetime,
           LEAD(inpatient_visit_earliest_start_datetime) OVER ({w_admission_order}) AS next_admission_start_datetime
    FROM er_join
    """
    spark.sql(query_er_inpatient1).createOrReplaceTempView("er_inpatient1")

    query_er_inpatient2 = f"""
    SELECT person_id, 
           admission_group, 
           inpatient_visit_earliest_start_datetime,
           inpatient_visit_end_datetime,
           prior_admission_end_datetime,
           length_of_admission_days,
           next_admission_start_datetime,
           {days_between("inpatient_visit_earliest_start_datetime", "prior_admission_end_datetime")} AS length_of_prior_gap_days,
           {days_between("next_admission_start_datetime", "inpatient_visit_end_datetime")} AS length_of_next_gap_days 
    FROM er_inpatient1
    """
    spark.sql(query_er_inpatient2).createOrReplaceTempView("er_inpatient2")

    query_er_inpatient3 = f"""
    SELECT person_id, 
           admission_group, 
           inpatient_visit_earliest_start_datetime,
           inpatient_visit_end_datetime,
           prior_admission_end_datetime,
           length_of_admission_days,
           next_admission_start_datetime,
           length_of_prior_gap_days,
           length_of_next_gap_days,
           CAST(length_of_prior_gap_days BETWEEN 0 AND {admission_gap} AS INT) AS prior_admission_within_30_days,
           CAST(length_of_next_gap_days BETWEEN 0 AND {admission_gap} AS INT) AS next_admission_within_30_days,
           LAG(length_of_admission_days) OVER ({w_admission_order}) AS length_of_prior_admission_days,
           COUNT(*) OVER ({w_admission_prior}) as number_of_prior_admissions
    FROM er_inpatient2
    """
    spark.sql(query_er_inpatient3).createOrReplaceTempView("er_inpatient3")

    query_latest = f"""
    SELECT MAX(visit_end_datetime) AS latest_datetime
    FROM visit_occurrence
    """
    spark.sql(query_latest).createOrReplaceTempView("latest")

    query_er_inpatient4 = f"""
    SELECT i.person_id, 
           i.admission_group, 
           i.inpatient_visit_earliest_start_datetime,
           i.inpatient_visit_end_datetime,
           i.length_of_admission_days,
           i.prior_admission_within_30_days,
           i.length_of_prior_gap_days,
           i.length_of_prior_admission_days,
           i.number_of_prior_admissions,
           i.next_admission_within_30_days
    FROM er_inpatient3 i
    CROSS JOIN latest l
    WHERE {days_between("l.latest_datetime", "i.inpatient_visit_end_datetime")} > {admission_gap}
    """
    er_to_inpatient4 = spark.sql(query_er_inpatient4)
    return er_to_inpatient4



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

    register_tables(spark, INPUT_TABLES)
    inpatient = inpatient_group(spark, INPATIENT_VISIT_CONCEPT_ID, EMERGENCY_VISIT_CONCEPT_ID, ER_LEAD_IN_THRESHOLD, ADMISSION_GAP)