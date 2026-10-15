# Readmission within 30 days

This project creates a pipeline on AWS to train a XGBoosy classifier to predict readmission within 30 days of inpatients using the Synthea-OMOP dataset, a synthetic dataset consisting of 100K patients in the OMOP Common Data Model. The model achieves an AUC of 0.91 is obtained with the test set.


## Project directory

```
.
├── config/
│   └── config.yaml
├── environment.yaml
└── scripts/
    ├── step1_copy_raw_data.sh
    ├── step2_create_glue_data_catalog.py
    ├── step3_convert_raw_to_parquet.py
    ├── step4_feature_engineering.py
    ├── step5_train.py
    ├── step6_evaluate.py
    └── glue_job.py
```


## Setup for AWS

Set up the following to run the project on AWS:
- AWS account
- AWS CLI
- S3 bucket
- IAM role for Glue with access to the S3 bucket, Glue Data catalog and CloudWatch logs


## Clone repository to local computer
Download repository to specified lcoation in local computer.
```bash
git clone https://github.com/egistific/readmission.git /path/to/project/directory/readmission
```

## Environment setup and activation

1) Install the conda environment
```bash
cd /path/to/project/directory/readmission
conda create --file environment.yaml

```
2) Activate the conda environment
```bash
conda activate glue_env
```


## Config file configuration

Edit `config/config.yaml` for the following items and change the values based on your account and preferences:

```yaml
aws:
  region: "us-east-1"

catalog:
  glue_role_arn: "arn:aws:iam::123456789012:role/MyGlueRole"
```

## Pipeline
1) Stream data from synthea-omop bucket, decompress and save the data as csv files in your own bucket.
```bash
sudo apt-get install lzop
cd scripts
bash step1_copy_raw_data.sh
```

2) Create Glue data datalog.
```bash
cd ../scripts
python step2_create_glue_data_catalog.py
```

3) Read tables from the catalog, set schemas and create parquet tables.
```bash
python glue_job.py \
  --job_name convert_raw_to_parquet \
  --script step3_convert_raw_to_parquet.py \
  --output_dir parquet
```

4) Read parquet tables and create features and label (duration of admission, whether there is previous admission within past 30 days, duration between previous discharge and start of inpatient visit, duration of previous admission, number of previous admissions, number of distinct conditions until end of inpatient visit, number of distinct drugs until end end of inpatient visit, number of distinct procedures until end of inpatient visit, the distinct conditions, distinct drugs and distinct conditions, readmission within 30 days).
```bash
python glue_job.py \
  --job_name create_features \
  --script step4_create_features.py \
  --input_dir parquet \
  --output_dir features
```

5) Train XGBoost classifer with the features from the earlier step using K-fold cross-validation. The data is randomly split into a training set and test set at patient-level and stratified on the labels. The training set used in stratfied group K-fold cross-validation. Perform paramter tuning using grid search to find the optimal parameters of the model, with AUC used as validation metric.
```bash
python glue_job_local.py \
  --job_name train_xgboost \
  --script step5_train_local.py \
  --input_dir features \
  --output_dir predictions
```

6) Evaluate the performance of the model using metrics using 0.5 decision threshold.
```bash
python glue_job_local.py \
  --job_name evaluate \
  --script step6_evaluate.py \
  --input_dir predictions \
  --output_dir metrics
```


## Outputs

```
s3://<bucket>/
├── <table>/                        # Step 1: raw CSV files
├── parquet/<table>/                # Step 3: raw parquet tables
├── features/                       # Step 4: feature tables
├── predictions/                   
│   ├── train/                      # Step 5: parquet table for best model's predictions on training set
│   ├── test/                       # Step 5: parquet table for best model's predictions on test set
│   └── best_model_metrics/         # Step 5: best parameters and average metric of K folds
├── metrics/
│   ├── train/{metrics.csv, cm.csv} # Step 6: metrics and confusion matrix for training set
│   └── test/{metrics.csv, cm.csv}  # Step 6: metrics and confusion matrix for test set
├── <scripts_name>/                 # uploaded job scripts
└── config/config.yaml              # uploaded config file
```


References:
- [Synthea] (https://github.com/synthetichealth/synthea) 
