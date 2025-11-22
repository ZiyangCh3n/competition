#!/usr/bin/env bash

# Usage:
#   ./extract_params.sh model.py > parameters.json

INPUT="$1"

if [ -z "$INPUT" ]; then
    echo "Usage: $0 <python_file>" >&2
    exit 1
fi

echo "{"

# Grep parameter assignments that look like:
#   NAME = value
# ignoring comments and function bodies.
grep -E '^[A-Z0-9_]+\s*=' "$INPUT" |
grep -v '^#' |
while IFS='=' read -r key value; do
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs)

    # Convert Python booleans → JSON booleans
    if [[ "$value" == "True" ]]; then
        value="true"
    elif [[ "$value" == "False" ]]; then
        value="false"
    fi

    # Convert Python None → JSON null
    if [[ "$value" == "None" ]]; then
        value="null"
    fi

    # Detect if numeric or string; quote only if non-numeric and not JSON boolean/null
    if ! [[ "$value" =~ ^-?[0-9]+(\.[0-9]+)?$ ]] &&
       ! [[ "$value" == "true" ]] &&
       ! [[ "$value" == "false" ]] &&
       ! [[ "$value" == "null" ]]; then
        value="\"$value\""
    fi

    echo "  \"$key\": $value,"
done | sed '$ s/,$//'   # remove trailing comma on last entry

echo "}"
