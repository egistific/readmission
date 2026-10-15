import os
import sys
import yaml
import logging

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from awsglue.dynamicframe import DynamicFrame
from awsglue.gluetypes import ChoiceType, ArrayType, StructType
from awsglue.transforms import ResolveChoice
from pyspark.context import SparkContext
from pyspark.sql import functions as F


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def find_ambiguous_columns(dyf):
    ambiguous_fields = []

    def traverse_schema(fields, current_path=""):
        for field in fields:
            field_path = f"{current_path}.{field.name}" if current_path else field.name

            if isinstance(field.dataType, ChoiceType):
                ambiguous_fields.append([field_path, [t for t in field.dataType.choices]])

            elif isinstance(field.dataType, StructType):
                traverse_schema(field.dataType.fields, field_path)

            elif isinstance(field.dataType, ArrayType):
                element_type = field.dataType.elementType
                if isinstance(element_type, ChoiceType):
                    ambiguous_fields.append([f"{field_path}[]", [t for t in element_type.choices]])
                elif isinstance(element_type, StructType):
                    traverse_schema(element_type.fields, f"{field_path}[]")

    traverse_schema(dyf.schema().fields)
    return ambiguous_fields


# Execute the function
def resolve_ambiguous_columns(dyf, exclude_suffixes, strategy="cast:string"):
    ambiguous_cols = find_ambiguous_columns(dyf)
    if ambiguous_cols:
        for path, types in ambiguous_cols:
            print(f"{path} has types {types}")

    resolved_cols = []

    for path, types in ambiguous_cols:
        if not any([path.endswith(s) for s in exclude_suffixes]):
            resolved_cols.append([path, strategy])
            
    if not resolved_cols:
        return dyf

    resolved_dyf = ResolveChoice.apply(frame=dyf, specs=resolved_cols)

    return resolved_dyf

def main():
    args = getResolvedOptions(sys.argv, ["CONFIG_PATH", "JOB_NAME", "OUTPUT_PREFIX"])

    config_filename = os.path.basename(args["CONFIG_PATH"])
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)
    
    sc = SparkContext.getOrCreate()
    glue_context = GlueContext(sc)
    
    job = Job(glue_context)
    job.init(args["JOB_NAME"], args)
    
    OUTPUT_PREFIX = args["OUTPUT_PREFIX"]
    YEAR_PARTITION_COLUMN = {"measurement": "measurement_date"}
    
    results = {}
    for table_name in config["table_names"]:
        catalog_table_name = f"{table_name}"
        dyf = glue_context.create_dynamic_frame.from_catalog(database=config["catalog"]["database_name"], table_name=catalog_table_name, transformation_ctx=f"read_{table_name}")
        clean_dyf = resolve_ambiguous_columns(dyf, config["exclude_suffixes"], strategy="cast:string")
        df = clean_dyf.toDF()

        select_exprs = []
        for col_name in df.columns:
            if col_name.endswith("concept_id"):
                select_exprs.append(F.col(col_name).cast("int").alias(col_name))
            elif col_name.endswith("_id"):
                select_exprs.append(F.col(col_name).cast("long").alias(col_name))
            elif col_name.endswith("_date"):
                select_exprs.append(F.col(col_name).cast("date").alias(col_name))
            elif col_name.endswith("_datetime"):
                select_exprs.append(F.col(col_name).cast("timestamp").alias(col_name))
            elif col_name.endswith("value_as_number"):
                select_exprs.append(F.col(col_name).cast("double").alias(col_name))
            else:
                select_exprs.append(F.col(col_name))
        
        df = df.select(*select_exprs)
            
        df_tagged = df.withColumn("omop_domain", F.lit(table_name))
        partition_keys = []
        if table_name in YEAR_PARTITION_COLUMN:
            source_col = YEAR_PARTITION_COLUMN[table_name]
            df_tagged = df_tagged.withColumn("year", F.year(F.col(source_col)))
            partition_keys = ["year"]

        dyf_out = DynamicFrame.fromDF(df_tagged, glue_context, f"{table_name}")
        output_path = f"{OUTPUT_PREFIX}/{table_name}/"
        connection_options = {"path": output_path, "purge": True}
        if partition_keys:
            connection_options["partitionKeys"] = partition_keys
        glue_context.write_dynamic_frame.from_options(
            frame=dyf_out,
            connection_type="s3",
            format="parquet",
            connection_options=connection_options,
            transformation_ctx=f"write_{table_name}"
        )
        partition_note = f" (partitioned by {partition_keys})" if partition_keys else ""
        print(f"Wrote Parquet to {output_path}{partition_note}")
        results[table_name] = "SUCCEEDED"
    job.commit()


if __name__ == "__main__":
    main()
