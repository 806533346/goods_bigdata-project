#!/bin/bash
# ============================================================
# COMP5434 Cloud - Teardown Script
# Terminates cloud resources to avoid ongoing charges.
#
# Usage:
#   bash scripts/teardown_cloud.sh
#   bash scripts/teardown_cloud.sh --keep-data   # preserve OSS/S3 data
# ============================================================
set -euo pipefail

KEEP_DATA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-data) KEEP_DATA=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "COMP5434 Cloud Teardown"
echo "============================================================"

# ── Stop Docker containers ──
echo "Stopping local Docker containers..."
docker compose -f docker/docker-compose.yml down 2>/dev/null || true
docker compose -f docker/docker-compose.spark.yml down 2>/dev/null || true

echo "Docker containers stopped."

# ── Cloud resource cleanup ──
echo ""
echo "Remember to manually terminate these cloud resources:"
echo "  1. GPU VM (ecs.gn6v / g4dn) — highest cost"
echo "  2. Spark worker VMs"
echo "  3. Spark master VM"
echo "  4. Prefect orchestrator VM (if created)"
echo "  5. Elastic IPs (if allocated)"

if [ "$KEEP_DATA" = false ]; then
    echo ""
    echo "Data cleanup (--keep-data NOT set):"
    echo "  6. Delete OSS bucket / S3 bucket"
    echo "  7. Delete ACR / ECR container images"
else
    echo ""
    echo "Data preserved (--keep-data set):"
    echo "  OSS/S3 data and container images retained."
fi

echo "============================================================"
echo "Teardown complete!"
echo "============================================================"
