# Readmission within 30 days

This project creates a pipeline on AWS to train an XGBoost classifier to predict readmission within 30 days of inpatients using the Synthea-OMOP dataset, a synthetic dataset consisting of 100K patients in the OMOP Common Data Model. The model achieves an AUC of 0.91 on the test set.
The code was tested on Ubuntu 24.04.4 LTS.

## Project directory

```
.
├── config
│   └── config.yaml
├── environment.yaml
├── README.md
└── scripts
    ├── glue_job.py
    ├── step1_copy_raw_data.sh
    ├── step2_create_glue_data_catalog.py
    ├── step3_convert_raw_to_parquet.py
    ├── step4_create_features.py
    ├── step5_train.py
    └── step6_evaluate.py

```

## Prerequisites
- AWS account
- AWS CLI
- S3 bucket
- IAM role for Glue with access to the S3 bucket, Glue Data catalog and CloudWatch logs
- Conda


## Clone repository to local computer
Clone repository to specified directory in local computer and go to project folder.
```bash
cd </path/to/specified/directory>
git clone https://github.com/egistific/readmission.git
cd readmission
```

## Environment setup and activation

1) Install the conda environment
```bash
conda env create -f environment.yaml
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
  database_name: "synthea-omop"
  bucket: "readmission-bucket"
  glue_role_arn: "arn:aws:iam::123456789012:role/MyGlueRole"
```

## Pipeline
1) Stream data from synthea-omop bucket, decompress and save the data as csv files in your own AWS S3 bucket.
```bash
sudo apt-get install lzop
cd scripts
bash step1_copy_raw_data.sh
```

2) Create Glue data catalog in AWS.
```bash
python step2_create_glue_data_catalog.py
```

3) Read tables from the catalog, set schemas and create parquet tables.
```bash
python glue_job.py \
  --job_name convert_raw_to_parquet \
  --script step3_convert_raw_to_parquet.py \
  --output_dir parquet
```

4) Read parquet tables and create features and label (duration of admission, whether there is previous admission within past 30 days, duration between previous discharge and start of inpatient visit, duration of previous admission, number of previous admissions, number of distinct conditions until end of inpatient visit, number of distinct drugs until end of inpatient visit, number of distinct procedures until end of inpatient visit, the distinct conditions, distinct drugs and distinct procedures, readmission within 30 days).
```bash
python glue_job.py \
  --job_name create_features \
  --script step4_create_features.py \
  --input_dir parquet \
  --output_dir features
```

5) Train XGBoost classifier with the features from the earlier step using K-fold cross-validation. The data is randomly split into a training set and test set at patient-level and stratified on the labels. The training set is used in stratified group K-fold cross-validation. Perform parameter tuning using grid search to find the optimal parameters of the model, with AUC used as validation metric.
```bash
python glue_job.py \
  --job_name train_xgboost \
  --script step5_train.py \
  --input_dir features \
  --output_dir predictions
```

6) Evaluate model performance using a 0.5 decision threshold.
```bash
python glue_job.py \
  --job_name evaluate \
  --script step6_evaluate.py \
  --input_dir predictions \
  --output_dir metrics
```


## Outputs
```
.
├── config
│   └── config.yaml
├── csv                                                                               # Step 1: raw CSV files
│   ├── condition_occurrence
│   │   └── condition_occurrence.csv
│   ├── drug_exposure
│   │   └── drug_exposure.csv
│   ├── observation
│   │   └── observation.csv
│   ├── person
│   │   └── person.csv
│   ├── procedure_occurrence
│   │   └── procedure_occurrence.csv
│   └── visit_occurrence
│       └── visit_occurrence.csv
├── features
│   └── part-00000-63b9a8b3-518f-478f-aee8-44e30345f1b7-c000.snappy.parquet          # Step 4: feature tables
├── metrics
│   ├── test                                                                         # Step 6: metrics and confusion matrix for test set
│   │   ├── cm.csv
│   │   └── metrics.csv
│   └── train                                                                        # Step 6: metrics and confusion matrix for training set
│       ├── cm.csv
│       └── metrics.csv
├── parquet                                                                          # Step 3: raw parquet tables
│   ├── condition_occurrence
│   │   ├── part-00000-c383618f-45d0-4918-a48f-02aee3d5924b-c000.snappy.parquet
│   │   ├── part-00000-f3881996-e0ba-4b96-9fbc-f8a756d005ea-c000.snappy.parquet
│   │   └── ...
│   ├── drug_exposure
│   │   ├── part-00000-9a0574bb-bf1d-4ec5-a788-799293fd29df-c000.snappy.parquet
│   │   ├── part-00001-9a0574bb-bf1d-4ec5-a788-799293fd29df-c000.snappy.parquet
│   │   └── ...
│   ├── observation
│   │   ├── part-00000-b4d6504d-4488-492b-9292-9306d2b2d9ef-c000.snappy.parquet
│   │   ├── part-00001-b4d6504d-4488-492b-9292-9306d2b2d9ef-c000.snappy.parquet
│   │   └── ...
│   ├── person
│   │   └── part-00000-bc364b88-8d77-4b54-a92a-e4e8c4514356-c000.snappy.parquet
│   ├── procedure_occurrence
│   │   ├── part-00000-97457907-5ddd-4740-b63d-e0e17c49e37b-c000.snappy.parquet
│   │   ├── part-00001-97457907-5ddd-4740-b63d-e0e17c49e37b-c000.snappy.parquet
│   │   └── ...
│   └── visit_occurrence
│       ├── part-00000-64e988b9-5efc-46b8-80b9-b01ef239b283-c000.snappy.parquet
│       ├── part-00001-64e988b9-5efc-46b8-80b9-b01ef239b283-c000.snappy.parquet
│       └── ...
├── predictions
│   ├── best_model_metrics                                                            # Step 5: best hyperparameters and mean cross-validation AUC on training set
│   │   ├── part-00000-d646b224-5307-4b52-9b4f-95ed0174950f-c000.json
│   │   └── part-00000-d646b224-5307-4b52-9b4f-95ed0174950f-c000.json
│   ├── test                                                                          # Step 5: parquet table for best model's predictions on test set
│   │   └── part-00000-054da3af-f566-473d-8723-7b714321105d-c000.snappy.parquet        
│   └── train                                                                         # Step 5: parquet table for best model's predictions on training set
│       └── part-00000-52750a14-7f1f-4dde-8457-2d914c3a47b8-c000.snappy.parquet       
└── scripts
    ├── glue_job.py
    ├── step1_copy_raw_data.sh
    ├── step2_create_glue_data_catalog.py
    ├── step3_convert_raw_to_parquet.py
    ├── step4_create_features.py
    ├── step5_train.py
    └── step6_evaluate.py

```



References:
- [Synthea-OMOP dataset](https://registry.opendata.aws/synthea-omop/) 
