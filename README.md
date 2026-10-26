# ONTHEDOT: Qwen football analytics

A small, runnable demonstration of how ONTHEDOT turns a football question into a validated analytics request. Type your own question and inspect the model output, normalized JSON, executed SQL and calculated result.

Qwen supplies the language understanding. Python validates the request and calculates statistics with SQLite. The model uses pretrained weights with a task prompt. This repository does not train or fine-tune Qwen.

```text
Your question -> Qwen -> JSON -> schema and grounding checks -> parameterized SQL -> result
```

The local demo uses the complete Premier League 2024-25 season already downloaded into ONTHEDOT: 380 matches, 20 clubs and 15,188 player records. Player statistics use the 11,567 records with positive minutes, covering 562 players. Original null values remain unknown.

The public repository includes the catalog and recorded model outputs. You can try question parsing immediately after setup. Calculated results and executed SQL require the separately supplied, authorized season export described below.

## Run locally

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.12 is selected by `.python-version`. Internet access is needed for the first dependency and model downloads. Run these commands in Terminal:

```bash
git clone https://github.com/landfvit/onthedot-qwen-demo.git
cd onthedot-qwen-demo
```

Choose one live runtime.

**Apple Silicon: MLX**

```bash
uv sync --locked --extra mlx
uv run --no-sync python run.py
```

The first live question downloads the pinned [Qwen3 4B Instruct MLX model](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit). Its weights are about 2.3 GB. Initial loading takes longer than subsequent questions. Later runs reuse the local model cache.

**Windows, Linux or macOS: Ollama**

Install and start [Ollama](https://ollama.com/download), then run:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
uv sync --locked
uv run --no-sync python run.py --backend ollama
```

This uses a separate [Qwen3 4B Instruct quantization](https://ollama.com/library/qwen3:4b-instruct-2507-q4_K_M), about 2.5 GB. Speed depends on the computer. Both runtimes process questions locally without an API key.

Open [localhost:8003](http://127.0.0.1:8003). Select **Live Qwen**, type a question and press **Run question**. Open **Available data** to see the supported names and statistics. An example dropdown is available, but live questions do not need to match an example.

Keep the Terminal process running while using the viewer. If port 8003 is busy, add `--port 8004` to the launch command and open localhost:8004.

Try: "Compare Haaland and Salah by assists in 2024-25." Qwen selects `player_comparison`, `assists`, both canonical names and the season. The local season export returns 3 assists for Haaland and 18 for Salah. You can also type a new question, such as "How many home points did Liverpool get in 2024-25?"

You can ask about monthly player trends, per-90 statistics, club points, wins and home or away results. The catalog covers this one PL season. Unknown players, ambiguous names and subjective questions need clarification. Use "Cole Palmer" rather than "Palmer", because the latter can identify two people in the source data.

## Season data

The full local export is `.local/real-season.json`. It is excluded from Git. Without it, Qwen can still parse your question, but statistics and executed SQL are unavailable. The viewer uses SQLite automatically. PostgreSQL is only needed when exporting from an existing ONTHEDOT database.

[API-Football's terms](https://www.api-football.com/terms) do not grant a publication license for its data, so the downloaded season is kept local. An employer needs an authorized copy of the export matching `data/catalog.json` to reproduce the full analytics demo.

Place that authorized export in the cloned project. On macOS or Linux:

```bash
mkdir -p .local
cp "/path/to/real-season.json" .local/real-season.json
```

In Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force .local
Copy-Item "C:\path\to\real-season.json" .local/real-season.json
```

Replace the example source path with your actual file location, then restart the viewer. The file's SHA-256 must match `dataset_id` in `data/catalog.json`. An unrelated JSON file or a different export will be rejected.

To export a complete season from an existing ONTHEDOT PostgreSQL database, use the optional export dependency:

```bash
uv sync --locked --extra export --extra mlx
uv run --no-sync python tools/export_season.py --database "dbname=your_database host=localhost" --season 2024
```

Replace `your_database` with the existing ONTHEDOT database name. This repository does not create or populate that source database. Stop the viewer before exporting. The command reads the source in a read-only transaction and makes no provider API calls. It checks all 380 fixtures and preserves player IDs and nulls. The MLX extra is kept installed on Apple Silicon and ignored on other platforms.

Every export replaces `data/catalog.json` and changes the fingerprint, including when only the export timestamp changes. Refresh the recordings on Apple Silicon using the pinned MLX model, then restart the viewer:

```bash
uv sync --locked --extra mlx
ONTHEDOT_BACKEND=mlx ONTHEDOT_MODEL_PATH= uv run --no-sync python tools/evaluate.py --mode live --record data/traces.json --output .local/evaluation-live.json
```

The recorder can save valid recordings and still exit with status 1 when any smoke case differs from its expected intent. Inspect the printed summary and evaluation report. If no compatible recordings are available, live question parsing remains usable.

## Preview without a model

```bash
uv sync --locked
uv run --no-sync python run.py
```

Choose **Recorded examples** to replay eleven genuine saved Qwen responses. This mode only supports the exact included questions. Each response still passes through validation. Analytics runs locally when the season file is available. Recorded inference times describe the original MLX run, excluding model loading.

## What the code demonstrates

- [Prompt and catalog](onthedot/prompt.py): explicit operations, statistics and handling of unsupported questions.
- [Model integration](onthedot/parser.py): lazy MLX loading, a pinned model revision and serialized local inference. Ollama is an optional portable runtime.
- [Validation](onthedot/schema.py): strict JSON, allowed fields, canonical names and checks that explicit entity and season filters survive parsing.
- [Analytics](onthedot/analytics.py): fixed SQL templates with bound parameters. The SQL panel shows the actual executed SELECT statements and their parameters.
- Missing data stays distinct from zero. Partial totals are labeled and incomplete rates remain unknown. Per-90 rates use total minutes rather than averaging individual match rates.

The backend rejects model-supplied SQL and calculated values. Invalid outputs stop before a database query. Grounding checks cover names and explicit filters, but do not prove that every selected metric or operation matches the user's meaning.

## Verification and limitations

```bash
uv sync --locked --extra dev --extra mlx
uv run --no-sync pytest -q
uv run --no-sync python tools/evaluate.py --mode recorded --output .local/evaluation.json
```

The [recordings](data/traces.json) preserve raw outputs, the model revision, prompt hash and season fingerprint. The full-season MLX run matched **9 of 11 curated smoke cases**. Ten outputs passed validation. The weather question was classified as `clarification` rather than `unsupported`. An instruction override produced a prose refusal instead of JSON and was blocked. Both failures remain visible. All seven analytical examples matched the expected request.

These cases illustrate behavior and failure handling. They are not a held-out benchmark or a claim about production accuracy. The evaluator exits with status 1 while any expected intent differs. Automated tests cover validation, parameterized SQL, weighted rates, missing inputs and read-only export behavior. The test command keeps MLX installed on Apple Silicon and works without the private season file.

Live MLX inference was verified against the full catalog. Processing its roughly 17,500 input tokens took about 17 to 19 seconds per example on the test computer, excluding initial loading. The Ollama adapter has HTTP contract tests, but its model has not been run as part of this demo's validation. Results can differ between quantizations. Ollama's displayed inference time includes the local request and may include loading.

## Screenshots

A new question interpreted by live Qwen, with the validated request and calculated club result.

![Live question and result](docs/images/live-question.jpg)

The JSON-to-SQL step exposes the executed statement and its bound parameters.

![Executed SQL](docs/images/sql-query.jpg)

An invalid model response is displayed and prevented from reaching analytics.

![Validation prevents execution](docs/images/validation.jpg)
