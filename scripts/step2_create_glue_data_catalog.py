import sys
import time
import boto3
import yaml
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

REGION = config["aws"]["region"]
BUCKET = config["catalog"]["bucket"]
DATABASE_NAME = config["catalog"]["database_name"]
GLUE_ROLE_ARN = config["catalog"]["glue_role_arn"]
CSV_NAME = config["catalog"]["csv_name"]
TABLE_NAMES = config["table_names"]

CRAWLER_NAME = DATABASE_NAME + "-crawler"

glue = boto3.client("glue", region_name=REGION)

# Create namespace for tables
try:
    glue.create_database(DatabaseInput={"Name": DATABASE_NAME})
except glue.exceptions.AlreadyExistsException:
    print(f"Database {DATABASE_NAME} already created!")


targets = {"S3Targets": [{"Path": f"s3://{BUCKET}/{CSV_NAME}/{t}/"} for t in TABLE_NAMES]} # crawler needs trailing slash

try:
    glue.create_crawler(Name=CRAWLER_NAME, Role=GLUE_ROLE_ARN, DatabaseName=DATABASE_NAME, Targets=targets, SchemaChangePolicy={"UpdateBehavior": "UPDATE_IN_DATABASE", "DeleteBehavior": "LOG"})
except glue.exceptions.AlreadyExistsException:
    glue.update_crawler(Name=CRAWLER_NAME, Role=GLUE_ROLE_ARN, DatabaseName=DATABASE_NAME, Targets=targets, SchemaChangePolicy={"UpdateBehavior": "UPDATE_IN_DATABASE", "DeleteBehavior": "LOG"})
    print(f"Crawler {CRAWLER_NAME} updated!")

glue.start_crawler(Name=CRAWLER_NAME)
print(f"Crawler {CRAWLER_NAME} started!")

try:
    while True:
        state = glue.get_crawler(Name=CRAWLER_NAME)["Crawler"]["State"]
        print(f"Crawler state: {state}")
        if state in ("RUNNING", "STOPPING"):
            seen_running = True
        if state == "READY" and seen_running:
            break
        time.sleep(30)
except KeyboardInterrupt:
    try:
        glue.stop_crawler(Name=CRAWLER_NAME)
        print("[-] Stop request sent to AWS. Exiting script.")
    except glue.exceptions.CrawlerStoppingException:
        print("[-] Crawler is already in the process of stopping.")
    except Exception as e:
        print(f"[!] Could not stop crawler via API: {e}")
    sys.exit(1)


# Check status of last crawl
last_crawl = glue.get_crawler(Name=CRAWLER_NAME)["Crawler"].get("LastCrawl", {})
if last_crawl.get("Status") == "FAILED":
    raise RuntimeError(f"Crawler run failed: {last_crawl.get('ErrorMessage')}")
print(f"Crawler run done: {last_crawl.get('Status')}")

discovered = glue.get_tables(DatabaseName=DATABASE_NAME)["TableList"]
print(f"Glue Data Catalog has {len(discovered)} tables in {DATABASE_NAME}: {[t['Name'] for t in discovered]}!")