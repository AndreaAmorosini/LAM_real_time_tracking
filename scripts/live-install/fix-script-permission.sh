#!/usr/bin/env bash
set -Eeuo pipefail

for file in scripts/*.sh; do
  [ -f "$file" ] && chmod +x "$file"
done

for file in scripts/live-install/*.sh; do
  [ -f "$file" ] && chmod +x "$file"
done

for file in scripts/live-install/setup/*.sh; do
  [ -f "$file" ] && chmod +x "$file"
done

for file in scripts/install/*.sh; do
  [ -f "$file" ] && chmod +x "$file"
done

echo "Script permissions fixed."
