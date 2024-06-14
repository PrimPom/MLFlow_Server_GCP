#!/bin/bash 

mlflow db upgrade $POSTGRESQL_URL
mlflow server \
  --host 10.27.0.27 \
  --port 8080 \
  --backend-store-uri $POSTGRESQL_URL \
  --artifacts-destination $STORAGE_URL