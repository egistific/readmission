#!/bin/bash
set -o pipefail

CONFIG=../config/config.yaml
[ -f "$CONFIG" ] || { echo "Config file not found: $CONFIG (run from scripts/)" >&2; exit 1; }

cfg() { python3 -c "import sys, yaml; d = yaml.safe_load(open(sys.argv[1])); print($1)" "$CONFIG"; }

BUCKET=$(cfg "d['catalog']['bucket']")     || exit 1
CSV_NAME=$(cfg "d['catalog']['csv_name']") || exit 1
readarray -t TABLES < <(cfg "'\n'.join(d['table_names'])")

[ -n "$BUCKET" ] && [ ${#TABLES[@]} -gt 0 ] || { echo "Empty bucket or table list in $CONFIG" >&2; exit 1; }

for t in "${TABLES[@]}"; do
  echo "decompressing: $t"

  if aws s3 cp --no-sign-request "s3://synthea-omop/synthea100k/${t}.csv.lzo" - \
      | lzop -d \
      | aws s3 cp - "s3://${BUCKET}/${CSV_NAME}/${t}/${t}.csv"; then
    echo "done: $t"
  else
    echo "FAILED: $t" >&2
    exit 1
  fi
done

echo "Finished copying files to S3 bucket!"