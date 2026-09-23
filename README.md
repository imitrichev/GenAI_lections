GenAI / LLM COURSE MATERIALS

CC BY / MIT LICENSE

## Laboratory work 1 - LanguageDetectorTool

[![Python tests](https://github.com/metablin/GenAI_lections/actions/workflows/python-app.yml/badge.svg)](https://github.com/metablin/GenAI_lections/actions/workflows/python-app.yml)
[![Coverage](https://raw.githubusercontent.com/metablin/GenAI_lections/coverage-badge/coverage.svg)](https://github.com/metablin/GenAI_lections/actions/workflows/python-app.yml)

Variant 28 implements `LanguageDetectorTool`, which detects the language of text
with `langdetect` and returns the most probable ISO language code together with
probability estimates.

Run the unit tests locally from `3_1_LLM_agent`:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

The existing Ollama integration tests are skipped by default. To run them:

```powershell
$env:RUN_INTEGRATION_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest -v
Remove-Item Env:RUN_INTEGRATION_TESTS
```
