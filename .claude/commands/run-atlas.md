# /run-atlas

Run a full scan session and write a RiskAtlas to disk.

## Usage

```
/run-atlas [--model <model_id>] [--domains <comma-list>] [--out <path>]
```

Defaults:
- `--model`: `gpt-4o`
- `--domains`: `science,history,law,medicine,finance,geography`
- `--out`: `atlas_<model_id>_<YYYYMMDD>.json`

## What this command does

1. Reads the relevant adapter from `src/hallucimap/models/`.
2. Instantiates one probe per domain in `--domains`.
3. Runs all probes asynchronously via `asyncio.gather`.
4. Feeds results into `HallucinationScorer`.
5. Builds / updates the `RiskAtlas` with the scored cells.
6. Serializes the atlas to `--out`.
7. Prints a summary table via `rich`.

## Prerequisite env vars

Set the relevant key before running:

```bash
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
```

## Example invocation (CLI equivalent)

```bash
hallucimap scan \
  --model claude-3-5-sonnet-20241022 \
  --domains science,history,medicine \
  --samples 5 \
  --output atlas_claude.json
```
