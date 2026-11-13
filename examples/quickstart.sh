#!/usr/bin/env bash
# Engram quickstart — store a few memories for one user, then ask about them.
# Usage: ENGRAM_API_KEY=your_key ./quickstart.sh
set -euo pipefail
: "${ENGRAM_API_KEY:?set ENGRAM_API_KEY to the key from your email}"
API=https://engram-c5m6.onrender.com/v1
USER_ID=${USER_ID:-demo-user}

remember() {
  curl -s -X POST "$API/ingest" -H "Content-Type: application/json" -H "X-API-Key: $ENGRAM_API_KEY" \
    -d "{\"user_id\": \"$USER_ID\", \"content\": \"$1\"}" > /dev/null
  echo "remembered: $1"
}

remember "I am training for a marathon in April"
remember "My left knee hurts after long runs"
remember "I switched to running in the mornings because evenings were too busy"

echo
echo "asking: what should I keep in mind when planning this user's training?"
curl -s -X POST "$API/query" -H "Content-Type: application/json" -H "X-API-Key: $ENGRAM_API_KEY" \
  -d "{\"user_id\": \"$USER_ID\", \"query\": \"what should I keep in mind when planning training?\", \"top_k\": 3}"
echo
