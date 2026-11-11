#!/bin/sh
# Copy the hooks in scripts/githooks into the directory git reads them from.
# Run it once per clone. Passing a hook name installs just that hook.
set -eu

repo_root=$(git rev-parse --show-toplevel)
source_dir="$repo_root/scripts/githooks"

if [ ! -d "$source_dir" ]; then
    echo "no hooks directory at $source_dir" >&2
    exit 1
fi

# core.hooksPath wins over .git/hooks when it is set, so install where git will look.
hooks_path=$(git config --get core.hooksPath || true)
if [ -n "$hooks_path" ]; then
    case "$hooks_path" in
        /*) hooks_dir="$hooks_path" ;;
        *) hooks_dir="$repo_root/$hooks_path" ;;
    esac
else
    hooks_dir="$repo_root/.git/hooks"
fi

mkdir -p "$hooks_dir"

if [ "$#" -gt 0 ]; then
    hooks=""
    for name in "$@"; do
        hooks="$hooks $source_dir/$name"
    done
else
    hooks="$source_dir"/*
fi

for hook in $hooks; do
    if [ ! -f "$hook" ]; then
        echo "no such hook: $hook" >&2
        exit 1
    fi
    name=$(basename "$hook")
    install -m 755 "$hook" "$hooks_dir/$name"
    echo "installed $hooks_dir/$name"
done

if ! command -v gitleaks >/dev/null 2>&1; then
    echo "note: gitleaks is not on PATH, so the pre-commit hook will refuse every commit until it is"
fi
