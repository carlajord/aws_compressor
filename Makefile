.PHONY: test lint format typecheck pipeline-local app

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src

pipeline-local:
	uv run python -m oidpipe.pipeline --local

app:
	uv run streamlit run app/Home.py --server.port 8501
