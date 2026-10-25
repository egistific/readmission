#!/bin/bash
set -o pipefail

YAML_FILE=../config/config.yaml

eval "$(python3 -c "
import yaml
import shlex

with open('$YAML_FILE', 'r') as f:
    config = yaml.safe_load(f)

print(f'BUCKET=\"{config[\"catalog\"][\"bucket\"]}\"')
print(f'CSV_NAME=\"{config[\"catalog\"][\"csv_name\"]}\"')

tables = ' '.join(shlex.quote(str(t)) for t in config['table_names'])
print(f'TABLE_NAMES=({tables})')
")"

for t in "${TABLE_NAMES[@]}"; do
  echo "Decompressing: $t"

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