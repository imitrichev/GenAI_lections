![coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fmamaelyaaa%2F04ceddcfc4dac22c10e87063d1130fea%2Fraw%2Fcoverage.json)

GenAI / LLM COURSE MATERIALS

CC BY / MIT LICENSE

### Быстрый старт через `uv` (рекомендуется)

```bash
# Установить зависимости (создаст .venv автоматически)
uv sync --dev

# Запустить пример LLM-агента
uv run python 3_1_LLM_agent/main.py

# Запустить тесты
uv run pytest -v
uv run pytest 3_1_LLM_agent/llm_agent_tests/test_tool_spell_checker.py -v

# Проверить покрытие
uv run pytest --cov --cov-report=term --cov-report=json
```

### Быстрый старт через `python` + `pip`

```shell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# Создать и активировать виртуальное окружение (Linux / WSL)
#python3 -m venv .venv
#source .venv/bin/activate
```

```shell
# Установка requirements.txt
pip install -r requirements.txt

# Установка dev-зависимости для тестов
pip install pytest pytest-cov
```

```shell
# Запустить LLMAgent
python 3_1_LLM_agent/main.py
```

```shell
# Запустить тесты
pytest -v
pytest 3_1_LLM_agent/llm_agent_tests/test_tool_spell_checker.py -v

# Проверить покрытие
pytest --cov --cov-report=term --cov-report=json
```