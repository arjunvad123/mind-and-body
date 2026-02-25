#!/bin/bash
# Download results from a completed Experiment 5 job
# Usage: ./get-results.sh

set -e

NAMESPACE="svcl-self-improve"
JOB_NAME="exp5-transformer-observer"
LOCAL_RESULTS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/cluster_results"

echo "=== Downloading Experiment 5 Results ==="

# Check job status
STATUS=$(kubectl get job "$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.conditions[0].type}' 2>/dev/null || echo "NotFound")
echo "Job status: $STATUS"

if [ "$STATUS" == "NotFound" ]; then
    echo "ERROR: Job $JOB_NAME not found in namespace $NAMESPACE"
    exit 1
fi

# Get pod name
POD=$(kubectl get pods -n "$NAMESPACE" -l "job-name=$JOB_NAME" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
echo "Pod: $POD"

if [ -z "$POD" ]; then
    echo "ERROR: No pod found for job $JOB_NAME"
    exit 1
fi

# Create local results directory
mkdir -p "$LOCAL_RESULTS"

# Copy results
echo "Copying results to $LOCAL_RESULTS/..."
kubectl cp "$NAMESPACE/$POD:/data/experiment5/results" "$LOCAL_RESULTS/" 2>/dev/null || true

# Copy training summary
kubectl cp "$NAMESPACE/$POD:/data/experiment5/results/probe_results.json" "$LOCAL_RESULTS/probe_results.json" 2>/dev/null || true
kubectl cp "$NAMESPACE/$POD:/data/experiment5/results/control_results.json" "$LOCAL_RESULTS/control_results.json" 2>/dev/null || true
kubectl cp "$NAMESPACE/$POD:/data/experiment5/results/training_summary.json" "$LOCAL_RESULTS/training_summary.json" 2>/dev/null || true

echo ""
echo "=== Results downloaded to $LOCAL_RESULTS/ ==="
ls -la "$LOCAL_RESULTS/" 2>/dev/null || echo "(empty)"
echo ""
echo "View results:"
echo "  cat $LOCAL_RESULTS/probe_results.json | python -m json.tool"
echo "  cat $LOCAL_RESULTS/control_results.json | python -m json.tool"
