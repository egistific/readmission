import os
import sys
import argparse
import time
import boto3
import yaml
import logging


logging.basicConfig(level=logging.INFO)

TERMINAL_STATES = {"SUCCEEDED", "FAILED", "STOPPED", "TIMEOUT"}

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", default="../config/config.yaml")
    ap.add_argument("--job_name", default="create_features")
    ap.add_argument("--script", default="local.py") # "glue_job_step1b_raw_to_parquet.py"
    ap.add_argument("--modules", nargs="*", default=[])
    ap.add_argument("--input_dir", default="parquet")
    ap.add_argument("--output_dir", default="features")
    ap.add_argument("--poll-seconds", type=int, default=15, help="Polling interval in seconds")
    return ap.parse_args(argv)    


def upload_files(s3, args, bucket, scripts_name):
    if args.modules:
        files = [args.config_path, args.script, *args.modules]
    else:
        files = [args.config_path, args.script]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        sys.exit(f"Files missing: {(', ').join(missing)}!")
    for f in files:
        if f.endswith(".yaml"):
            key = f"config/{os.path.basename(f)}"
        else:
            key =  f"{scripts_name}/{f}"
        s3.upload_file(f, bucket, key)
        print(f"Uploaded {f} --> s3://{bucket}/{key}!")


def build_job(args, glue_role_arn, bucket, scripts_name, glue_version, worker_type, num_workers, timeout):
    def get_uri(name):
        if name.endswith(".yaml"):
            return f"s3://{bucket}/config/{os.path.basename(name)}"
        return f"s3://{bucket}/{scripts_name}/{os.path.basename(name)}"
    default_args = {
        "--job_language": "python",
        "--enable-metrics": "true",
        "--enable-continuous-cloudwatch-log": "true",
        "--additional-python-modules": "xgboost==2.0.3,scikit-learn>=1.0.0,PyYAML==6.0.3" # modules needed to train xgboost model
    }
    yaml_files = [get_uri(args.config_path)]
    if args.modules:
        py_files = [get_uri(m) for m in args.modules if m.endswith(".py")]
        yaml_files = yaml_files + [get_uri(m) for m in args.modules if m.endswith(".yaml")]
        if len(py_files) > 0:
            default_args["--extra-py-files"] = (",").join(py_files)
    default_args["--extra-files"] = (",").join(yaml_files)
    return dict(
        Role=glue_role_arn,
        Command={"Name": "glueetl", "ScriptLocation": get_uri(args.script), "PythonVersion": "3"},
        DefaultArguments=default_args,
        GlueVersion=str(glue_version),
        WorkerType=worker_type,
        NumberOfWorkers=int(num_workers),
        Timeout=int(timeout),
        MaxRetries=0
    )

def create_or_update(glue, job_name, job):
    try:
        glue.create_job(Name=job_name, **job)
        print(f"Created job {job_name}!")
    except glue.exceptions.AlreadyExistsException:
        glue.update_job(JobName=job_name, JobUpdate=job)
        print(f"Job {job_name} updated!")
    except glue.exceptions.IdempotentParameterMismatchException:
        glue.update_job(JobName=job_name, JobUpdate=job)
        print(f"Job {job_name} updated!")

def wait_for(glue, job_name, run_id, poll):
    t0 = time.time()
    last = None
    try:
        while True:
            run = glue.get_job_run(JobName=job_name, RunId=run_id)["JobRun"]
            state = run["JobRunState"]
            if state != last:
                print(f"[{int(time.time() - t0):>4}s] {state}")
                last = state
            if state in TERMINAL_STATES:
                return run      
            time.sleep(poll)
    except KeyboardInterrupt:
        try:
            print("\nInterrupted Glue run so it does not keep billing!")
            glue.batch_stop_job_run(JobName=job_name, JobRunIds=[run_id])
            verify_reponse = glue.get_job_run(JobName=job_name, RunId=run_id)
            print(f"Updated AWS job state: {verify_reponse['JobRun']['JobRunState']}!")
        except Exception as e:
            print(f"Failed to cancel AWS glue job: Error: {e}!")
        sys.exit(1)

def show_error_log(region, run_id):
    try:
        logs = boto3.client("logs", region_name=region)
        events = logs.get_log_events(logGroupName="/aws-glue/jobs/error", logStreamName=run_id, limit=40, startFromHead=False)["events"]
        print("\nLines of the job's error log:")
        for e in events:
            print(f" {e['message'].rstrip()}")
    except:
        print(f"\nCould not read log. Open CloudWatch > Log groups > /aws-glue/jobs/error or /aws-glue/jobs/logs-v2 and look for run id {run_id}!")


def check_output(s3, output_uri):
    bucket, _, prefix = output_uri.replace("s3://", "", 1).partition("/")
    n = size = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            n += 1
            size += obj["Size"]
    if n:
        print(f"Output: {n} parquet files {size / 1e6:.1f} MB at {output_uri}")
    else:
        print(f"Warning: Run succeeded but no parquet files were found in {output_uri}!")
    return n

def main(argv=None):

    args = parse_args(argv)
    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    REGION = config["aws"]["region"]
    BUCKET = config["catalog"]["bucket"]
    GLUE_ROLE_ARN = config["catalog"]["glue_role_arn"]

    SCRIPTS_NAME = config['scripts_name']
    INPUT_PREFIX = f"s3://{BUCKET}/{args.input_dir}"
    OUTPUT_PREFIX = f"s3://{BUCKET}/{args.output_dir}"

    GLUE_VERSION = config["job"]["glue_version"]
    WORKER_TYPE = config["job"]["worker_type"]
    NUM_WORKERS = config["job"]["num_workers"]
    TIMEOUT = config["job"]["timeout"]

    session = boto3.Session(region_name=REGION)
    glue, s3 = session.client("glue"), session.client("s3")

    job = build_job(args, GLUE_ROLE_ARN, BUCKET, SCRIPTS_NAME, GLUE_VERSION, WORKER_TYPE, NUM_WORKERS, TIMEOUT)
    run_args = {
        "--CONFIG_PATH": args.config_path,
        "--JOB_NAME": args.job_name,
        "--INPUT_PREFIX": str(INPUT_PREFIX),
        "--OUTPUT_PREFIX": str(OUTPUT_PREFIX),
        }
    upload_files(s3, args, BUCKET, SCRIPTS_NAME)
    create_or_update(glue, args.job_name, job)

    print(f"Starting job run for {args.job_name}...")
    run_id = glue.start_job_run(JobName=args.job_name, Arguments=run_args)["JobRunId"]
    run = wait_for(glue, args.job_name, run_id, args.poll_seconds)
    state = run["JobRunState"]
    
    print(f"\nFinished!")
    if state == "SUCCEEDED":
        check_output(s3, OUTPUT_PREFIX)
    else:
        show_error_log(REGION, run_id)


if __name__ == "__main__":
    main()
