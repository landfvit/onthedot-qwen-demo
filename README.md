# ONTHEDOT: Qwen football analytics

A small, runnable demonstration of how ONTHEDOT turns a football question into a validated analytics request. Type your own question and inspect the model output, normalized JSON, executed SQL and calculated result.

Qwen supplies the language understanding. Python validates the request and calculates statistics with SQLite. The model uses pretrained weights with a task prompt. This repository does not train or fine-tune Qwen.

```text
Your question -> Qwen -> JSON -> schema and grounding checks -> parameterized SQL -> result
```

The repository includes the complete Premier League 2024-25 season already downloaded into ONTHEDOT: 380 matches, 20 clubs and 15,188 player records. Player statistics use the 11,567 records with positive minutes, covering 562 players. Original null values remain unknown.

## Run locally without a model

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.12 is selected by `.python-version`. Run these commands in Terminal:

```bash
git clone https://github.com/landfvit/onthedot-qwen-demo.git
cd onthedot-qwen-demo
uv sync --locked
uv run --no-sync python run.py
```

Open [localhost:8003](http://127.0.0.1:8003). Choose **Recorded examples**, select a question and press **Run question**. Eleven genuine saved Qwen responses are available. Each response is validated. Valid analytical requests run against the bundled season, showing the JSON, executed SQL, parameters and calculated result. No database setup, data copy or model download is required.

Recorded mode supports only the included questions. Its inference times describe the original MLX run, excluding model loading. To write your own questions, install one live runtime below.

Keep the Terminal process running while using the viewer. If port 8003 is busy, add `--port 8004` to the launch command and open localhost:8004. Internet access is needed for the first dependency and model downloads.

## Ask your own questions

**Apple Silicon: MLX**

Stop the running viewer, then run:

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

Select **Live Qwen**, type a question and press **Run question**. Open **Available data** to see the supported names and statistics. Live questions do not need to match an example.

Try: "Compare Haaland and Salah by assists in 2024-25." Qwen selects `player_comparison`, `assists`, both canonical names and the season. The bundled season returns 3 assists for Haaland and 18 for Salah. You can also type a new question, such as "How many home points did Liverpool get in 2024-25?"

You can ask about monthly player trends, per-90 statistics, club points, wins and home or away results. The catalog covers this one PL season. Unknown players, ambiguous names and subjective questions need clarification. Use "Cole Palmer" rather than "Palmer", because the latter can identify two people in the source data.

## Season data

The default dataset is [data/season.json](data/season.json), exported from ONTHEDOT's API-Football data. It covers 16 August 2024 through 25 May 2025. The viewer creates SQLite in memory automatically. No PostgreSQL server or provider API key is needed to run the demo.

The project owner confirmed the rights and approved publication of this snapshot, catalog and Qwen examples. See [NOTICE](NOTICE) for provenance and the scope of that permission. This does not grant a broader data license.

The season's SHA-256 is `b4244d2af992777dee96126b0fa3cf51d66a117b66373488c3f294a39c92a83d`. The loader checks it against `dataset_id` in [data/catalog.json](data/catalog.json) before analytics can run, keeping the model catalog and results tied to the same snapshot.

### Optional: export from your own ONTHEDOT database

This step is for an existing ONTHEDOT PostgreSQL installation. Normal demo users can skip it. Stop the viewer, then run:

```bash
uv sync --locked --extra export --extra mlx
uv run --no-sync python tools/export_season.py --database "dbname=your_database host=localhost" --season 2024
```

Replace `your_database` with the source database name. This repository does not create or populate that database. The command reads it in a read-only transaction and makes no provider API calls. It checks all 380 fixtures, preserves player IDs and nulls, and writes `data/season.json` and `data/catalog.json` by default.

Every export changes the fingerprint, including when only the export timestamp changes. The previous recordings will no longer match. Refresh them on Apple Silicon using the pinned MLX model, then restart the viewer:

```bash
uv sync --locked --extra mlx
ONTHEDOT_BACKEND=mlx ONTHEDOT_MODEL_PATH= uv run --no-sync python tools/evaluate.py --mode live --record data/traces.json --output .local/evaluation-live.json
```

The recorder requires MLX and can exit with status 1 while saving valid recordings when a smoke case differs from its expected intent. Inspect the summary and evaluation report. Live parsing remains usable without compatible recordings.

## What the code demonstrates

- [Prompt and catalog](onthedot/prompt.py): explicit operations, statistics and handling of unsupported questions.
- [Model integration](onthedot/parser.py): lazy MLX loading, a pinned model revision and serialized local inference. Ollama is an optional portable runtime.
- [Validation](onthedot/schema.py): strict JSON, allowed fields, canonical names and checks that explicit entity and season filters survive parsing.
- [Analytics](onthedot/analytics.py): fixed SQL templates with bound parameters. The SQL panel shows the actual executed SELECT statements and their parameters.
- Missing data stays distinct from zero. Partial totals are labeled and incomplete rates remain unknown. Per-90 rates use total minutes rather than averaging individual match rates.

The `?` characters in SQL are placeholders. **Bound parameters** lists their values in the same order.

The backend rejects model-supplied SQL and calculated values. Invalid outputs stop before a database query. Grounding checks cover names and explicit filters, but do not prove that every selected metric or operation matches the user's meaning.

## Verification and limitations

```bash
uv sync --locked --extra dev --extra mlx
uv run --no-sync pytest -q
uv run --no-sync python tools/evaluate.py --mode recorded --output .local/evaluation.json
```

The [recordings](data/traces.json) preserve raw outputs, the model revision, prompt hash and season fingerprint. The full-season MLX run matched **9 of 11 curated smoke cases**. Ten outputs passed validation. The weather question was classified as `clarification` rather than `unsupported`. An instruction override produced a prose refusal instead of JSON and was blocked. Both failures remain visible. All seven analytical examples matched the expected request.

These cases illustrate behavior and failure handling. They are not a held-out benchmark or a claim about production accuracy. The evaluator exits with status 1 while any expected intent differs. Automated tests cover validation, parameterized SQL, weighted rates, missing inputs and read-only export behavior. The test command keeps MLX installed on Apple Silicon. Tests also work without a model.

Live MLX inference was verified against the full catalog. Processing its roughly 17,500 input tokens took about 17 to 19 seconds per example on the test computer, excluding initial loading. The Ollama adapter has HTTP contract tests, but its model has not been run as part of this demo's validation. Results can differ between quantizations. Ollama's displayed inference time includes the local request and may include loading.

## Screenshots

A new question interpreted by live Qwen, with the validated request and calculated club result.

![Live question and result](docs/images/live-question.jpg)

The JSON-to-SQL step exposes the executed statement and its bound parameters.

![Executed SQL](docs/images/sql-query.jpg)

An invalid model response is displayed and prevented from reaching analytics.

![Validation prevents execution](docs/images/validation.jpg)
