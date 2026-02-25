#!/bin/bash
# Deploy Experiment 6 to Nautilus cluster
# Usage: ./deploy.sh [--clean]

set -e

NAMESPACE="svcl-self-improve"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
JOB_NAME="exp6-liquid-observer"

echo "=== Experiment 6: Nautilus Deployment ==="
echo "Namespace: $NAMESPACE"
echo "Code dir:  $CODE_DIR"
echo ""

# ── Clean up previous run if requested ──────────────────────────────
if [ "$1" == "--clean" ]; then
    echo "Cleaning up previous deployment..."
    kubectl delete job "$JOB_NAME" -n "$NAMESPACE" 2>/dev/null || true
    kubectl delete configmap exp6-code -n "$NAMESPACE" 2>/dev/null || true
    echo "Cleaned."
    echo ""
fi

# ── Step 1: Create PVC (idempotent — won't fail if exists) ─────────
echo "Step 1: Creating PVC..."
kubectl apply -f "$SCRIPT_DIR/pvc-exp6-data.yaml" -n "$NAMESPACE"
echo ""

# ── Step 2: Create ConfigMap from Python source files ──────────────
echo "Step 2: Creating ConfigMap from experiment code..."
kubectl delete configmap exp6-code -n "$NAMESPACE" 2>/dev/null || true

kubectl create configmap exp6-code \
    --from-file=__init__.py="$CODE_DIR/__init__.py" \
    --from-file=config.py="$CODE_DIR/config.py" \
    --from-file=dynamical_systems.py="$CODE_DIR/dynamical_systems.py" \
    --from-file=executor_model.py="$CODE_DIR/executor_model.py" \
    --from-file=dataset.py="$CODE_DIR/dataset.py" \
    --from-file=observer_model.py="$CODE_DIR/observer_model.py" \
    --from-file=trainer.py="$CODE_DIR/trainer.py" \
    --from-file=dynamics_analysis.py="$CODE_DIR/dynamics_analysis.py" \
    --from-file=probes.py="$CODE_DIR/probes.py" \
    --from-file=controls.py="$CODE_DIR/controls.py" \
    --from-file=run_experiment.py="$CODE_DIR/run_experiment.py" \
    -n "$NAMESPACE"
echo ""

# ── Step 3: Delete old job if it exists ────────────────────────────
echo "Step 3: Cleaning up old job (if any)..."
kubectl delete job "$JOB_NAME" -n "$NAMESPACE" 2>/dev/null || true
sleep 2
echo ""

# ── Step 4: Submit the GPU job ─────────────────────────────────────
echo "Step 4: Submitting GPU job..."
kubectl apply -f "$SCRIPT_DIR/job-exp6.yaml" -n "$NAMESPACE"
echo ""

# ── Step 5: Monitor ────────────────────────────────────────────────
echo "=== Deployment complete! ==="
echo ""
echo "Monitor with:"
echo "  kubectl get pods -n $NAMESPACE -w"
echo "  kubectl logs -f job/$JOB_NAME -n $NAMESPACE"
echo ""
echo "Check status:"
echo "  kubectl get jobs -n $NAMESPACE"
echo ""
echo "Download results when done:"
echo "  POD=\$(kubectl get pods -n $NAMESPACE -l job-name=$JOB_NAME -o jsonpath='{.items[0].metadata.name}')"
echo "  kubectl cp $NAMESPACE/\$POD:/data/experiment6/results ./results"
echo ""
echo "Clean up when done:"
echo "  kubectl delete job $JOB_NAME -n $NAMESPACE"
