#!/bin/bash
# Run on BOTH GPU NODES
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
MY_IP=$(hostname -I | awk '{print $1}')
pkill -f "spark.deploy.worker" 2>/dev/null || true
sleep 1

nohup env JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
  /usr/local/bin/spark-class org.apache.spark.deploy.worker.Worker \
  spark://10.0.1.159:7077 \
  --host $MY_IP --cores 4 --memory 16g \
  > /app/output/spark_worker.log 2>&1 &

sleep 2
echo "Worker started on $MY_IP ✓"
