set positional-arguments

default:
    @just --list

setup:
    uv sync --locked
    uvx --from=prek==0.5.6 prek install

check:
    uvx --from=prek==0.5.6 prek run --all-files

test *args:
    uv run --locked --package=forecasters --group=dev pytest "$@"

test-native *args:
    FORECASTERS_NATIVE=1 uv run --locked --package=forecasters --group=dev --group=native pytest packages/forecasters/tests/integration "$@"

build:
    uv build --package=forecasters --no-sources --out-dir=dist/forecasters
