#!/usr/bin/env bash
# Usage: ./scripts/push_to_github.sh https://github.com/<you>/ecom-fl-demand-supply.git
# Create the EMPTY repo on github.com first (no README/license), then run this.
set -euo pipefail
REMOTE="${1:?pass your repo URL}"
git init -q 2>/dev/null || true
git add -A
git commit -qm "Federated demand forecasting & inventory replenishment with Flower" || true
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE"
git push -u origin main
