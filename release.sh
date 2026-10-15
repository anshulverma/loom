#!/usr/bin/env bash
# Build, publish to TestPyPI then PyPI, verify each install, and tag the release.
# Usage: TEST_TOKEN=... TOKEN=... ./release.sh   (on a devserver: with-proxy ./release.sh)
# Safe to re-run after a partial failure: files already uploaded are skipped.
set -euo pipefail
cd "$(dirname "$0")"

: "${TEST_TOKEN:?set TEST_TOKEN to a test.pypi.org API token}"
: "${TOKEN:?set TOKEN to a pypi.org API token}"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "working tree is dirty; commit first" >&2
    exit 1
fi

version="$(uv version --short)"
echo "==> releasing loom-torch $version"

rm -rf dist
uv build

# Install loom-torch==$version from an index into a fresh venv and import it.
# Retries because a fresh upload can take a minute to show up in the index.
verify() {
    local venv
    venv="$(mktemp -d)"
    uv venv -q "$venv" -p 3.12
    for _ in {1..10}; do
        if uv pip install -q --no-cache -p "$venv/bin/python" "$@" "loom-torch==$version"; then
            "$venv/bin/python" -I -W ignore -c "import loom; print('import ok:', loom.__file__)"
            rm -rf "$venv"
            return
        fi
        echo "not installable yet, retrying in 15s"
        sleep 15
    done
    echo "could not install loom-torch==$version" >&2
    exit 1
}

echo "==> TestPyPI"
UV_PUBLISH_TOKEN="$TEST_TOKEN" uv publish \
    --publish-url https://test.pypi.org/legacy/ --check-url https://test.pypi.org/simple/
# torch is not on TestPyPI; uv tries --extra-index-url first, so dependencies come from PyPI.
verify --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/

echo "==> PyPI"
UV_PUBLISH_TOKEN="$TOKEN" uv publish --check-url https://pypi.org/simple/
verify --index-url https://pypi.org/simple/

echo "==> tagging v$version"
git rev-parse -q --verify "refs/tags/v$version" >/dev/null || git tag "v$version"
git push
git push origin "v$version"

echo "==> done: https://pypi.org/project/loom-torch/$version/"
