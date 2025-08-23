# Repository Guidelines

## Project Structure & Modules
- `mcpo_kcd/`: Python package (MCP server + LangGraph pipeline)
  - `mcp_server.py`: FastMCP stdio server exposing `kcd_query`.
  - `graph.py`, `pipeline.py`, `retriever.py`, `llm.py`, `utils.py`: core flow, fallback, retrieval, LLM, helpers.
- `data/`: domain text sources (`kcd_kb_*.txt`).
- `config.json`: mcpo config to launch this MCP server.
- `.env`: runtime configuration (see README).
- `requirements.txt`, `README.md`: deps and overview.

## Build, Test, and Development
- Install: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run via mcpo (recommended): `mcpo --config ./config.json`
  - Opens OpenAPI at `http://localhost:8000/kcd/docs` (tool: `kcd_query`).
- Direct run (debug): `python -m mcpo_kcd.mcp_server`
  - Uses FastMCP stdio; best used through `mcpo` for HTTP.
- Useful env vars (in `.env`): `DATA_DIR`, `MODEL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `EMBEDDING_MODEL`, `MCPO_REQUIRE_LLM`, `MCPO_REQUIRE_EMBEDDINGS`.

## Coding Style & Naming
- Python 3.10+; 4-space indentation; type hints preferred.
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.
- Keep files small and cohesive; colocate helpers in `utils.py`.
- Logging: use `logging` (level via `MCPO_LOG_LEVEL`). Avoid `print` in library code.

## Testing Guidelines
- Current: no formal pytest suite. Add targeted unit tests near modules when contributing.
- Smoke test locally:
  - Start server: `mcpo --config ./config.json`
  - Open docs: `http://localhost:8000/kcd/docs` and invoke `kcd_query` with a short query; verify category results.
- For changes in retrieval/LLM, test with and without network by toggling `MCPO_REQUIRE_*`.

## Commit & Pull Requests
- Commits: concise, present tense, scoped (e.g., "retriever: improve hybrid scoring").
- PRs must include:
  - Summary of change and rationale.
  - Affected modules/paths (e.g., `mcpo_kcd/retriever.py`).
  - Test plan (commands, screenshots of docs call if UI-facing).
  - Linked issues (if any).

## Security & Configuration
- Never commit secrets; prefer `.env` (gitignored).
- Large models/network are optional; ensure graceful fallback paths remain intact.
