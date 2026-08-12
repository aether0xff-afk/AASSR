#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-aether0xff-afk/AASSR}"
SOURCE_DIR="${2:-wiki}"
TOKEN="${WIKI_TOKEN:-${GITHUB_TOKEN:-}}"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory not found: $SOURCE_DIR" >&2
  exit 2
fi

if [[ -z "$TOKEN" ]]; then
  echo "Set WIKI_TOKEN (recommended) or GITHUB_TOKEN before running." >&2
  exit 2
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
WIKI_DIR="$TMP_DIR/wiki"
WIKI_URL="https://x-access-token:${TOKEN}@github.com/${REPO}.wiki.git"

if ! git ls-remote "$WIKI_URL" HEAD >/dev/null 2>&1; then
  cat >&2 <<'EOF'
GitHub Wiki is not initialized yet.
Open the repository's Wiki tab and create the first Home page once, then rerun this script.
EOF
  exit 3
fi

git clone "$WIKI_URL" "$WIKI_DIR"
find "$WIKI_DIR" -mindepth 1 -maxdepth 1 -type f -name '*.md' -delete
cp "$SOURCE_DIR"/*.md "$WIKI_DIR"/

cd "$WIKI_DIR"
git config user.name "AASSR Wiki Sync"
git config user.email "aether0xff@gmail.com"
git add -A

if git diff --cached --quiet; then
  echo "Wiki is already up to date."
  exit 0
fi

git commit -m "Sync AASSR research wiki"
git push origin HEAD

echo "Wiki synchronized: https://github.com/${REPO}/wiki"
