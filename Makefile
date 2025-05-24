format:
	@echo "🖨️ Format code: Running ruff"
	@uvx ruff format

test:
	@echo "🔎 Testing code: Running pytest"
	@uv run python -m pytest --cov=src

pack:
	@echo "🗂️ Packaging code into flatfile - use as knowledge base for Claude/aider/etc."
	@uvx repopack "$(CURDIR)" --ignore *lock*,.github/*,.mypy_cache/*,architecture-diagram*,*.svg --output "codebase.txt"

mypy:
	@uv run mypy "$(CURDIR)"
