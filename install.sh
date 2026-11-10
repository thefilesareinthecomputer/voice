#!/bin/sh
# Install voice from this folder, pinned to the package versions in uv.lock (the ones the tests
# and the vulnerability audit use), then run the guided setup. Run it again after `git pull`
# to update.
set -eu
CDPATH= cd -- "$(dirname -- "$0")"

if [ "$(uname -s)-$(uname -m)" != "Darwin-arm64" ]; then
    echo "voice needs a Mac with Apple Silicon (M1 or newer)." >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "voice needs uv. Install it with: brew install uv" >&2
    exit 1
fi

uv export --frozen --no-dev --no-emit-project --output-file requirements.txt
uv tool install --reinstall-package voice --constraints requirements.txt .

bin="$(uv tool dir --bin)"
case ":$PATH:" in
    *":$bin:"*) ;;
    *) echo "Note: $bin is not on your PATH yet. Run 'uv tool update-shell', then open a new terminal." >&2 ;;
esac
"$bin/voice" setup
