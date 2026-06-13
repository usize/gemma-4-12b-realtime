#!/bin/bash

set -eux

# Ensure a prompt was provided
if [ -z "$1" ]; then
    echo "Error: Please provide a research prompt."
    echo "Usage: $0 \"Your research topic or query\""
    exit 1
fi

PROMPT="$1"

# Run Claude optimized for cheap web research:
# 1. --model haiku enforces the cost-effective model
# 2. --allowedTools websearch locks the model into ONLY browsing the web
claude --print "please use websearch to research and return a report, here's your prompt: $PROMPT" \
  --model haiku \
  --allowedTools websearch \
  --dangerously-skip-permissions

