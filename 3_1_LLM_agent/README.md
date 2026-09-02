# LLM Agent

![Tests](https://github.com/bakost/GenAI_lections/actions/workflows/python-tests.yml/badge.svg)
![Coverage](https://raw.githubusercontent.com/bakost/GenAI_lections/main/3_1_LLM_agent/coverage.svg)

Простой LLM-агент с инструментами (калькулятор, веб-поиск) и аудит-логированием.

## AuditLogger

Класс [`AuditLogger`](llm_agent/tool_auditlogger.py) логирует все действия
агента — входящие запросы, планы, результаты выполнения инструментов и
финальные ответы — в структурированном JSON-формате для последующего аудита.

## Запуск тестов

Юнит-тесты `AuditLogger` (быстрые, без внешних зависимостей):

```bash
pip install -r requirements.txt
coverage run -m pytest tests/test_tool_auditlogger.py -v
coverage report -m
```

Интеграционные тесты `LLMAgent` с реальной моделью в Ollama (требуют
запущенный `ollama serve`). Локально по умолчанию используется компактная
`qwen3:0.6b` — быстро качается и быстро отвечает даже на CPU:

```bash
ollama pull qwen3:0.6b
python3 -m pytest tests/test_llm_agent_ollama_integration.py -v
```

Модель можно переопределить переменной окружения `OLLAMA_TEST_MODEL`,
например для локальной проверки на более крупной модели:

```bash
OLLAMA_TEST_MODEL=qwen3.5 python3 -m pytest tests/test_llm_agent_ollama_integration.py -v
```

В CI (GitHub Actions) интеграционные тесты гоняются в отдельном job'е
`ollama-integration` на модели `qwen3.5` — он сам поднимает Ollama на
раннере и кеширует скачанную модель между запусками, чтобы не качать
6.6 ГБ заново на каждый push.
