#!/bin/bash
# Run on CONTROL NODE (ssh root@<控制节点公网IP>)
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
pkill -f "spark.deploy.master" 2>/dev/null || true
sleep 1

nohup env JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
  /usr/local/bin/spark-class org.apache.spark.deploy.master.Master \
  --host <控制节点内网IP> --port 7077 --webui-port 8080 \
  > /app/output/spark_master.log 2>&1 &

sleep 4
curl -s http://<控制节点内网IP>:8080 | grep -q ALIVE && echo "Master ALIVE ✓ spark://<控制节点内网IP>:7077" || echo "FAILED - check /app/output/spark_master.log"
