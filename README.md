GenAI / LLM COURSE MATERIALS

CC BY / MIT LICENSE

---

## Лабораторная работа 1 — вариант 9: `QRCodeTool`

![QRCodeTool tests](https://github.com/nedreyner/GenAI_lections/actions/workflows/qrcode-tool.yml/badge.svg)
![Code Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/nedreyner/GIST_ID/raw/GenAI_lections_qrcode_coverage.json)

Инструмент [`QRCodeTool`](3_1_LLM_agent/llm_agent/tool_qrcode.py) генерирует QR-код
из текста и либо сохраняет его в файл, либо возвращает base64-строку с PNG.
Инструмент подключён к агенту в `3_1_LLM_agent/llm_agent/core_v2.py` под именем `qrcode`.

Формат запроса: `текст | путь/к/файлу.png`. Если путь не указан, возвращается base64.

```python
from llm_agent.tool_qrcode import QRCodeTool

tool = QRCodeTool()
print(tool.use("https://example.com | qr.png"))   # сохранит файл
print(tool.use("https://example.com"))            # вернёт base64-строку
```

Тесты: [`3_1_LLM_agent/tests/test_tool_qrcode.py`](3_1_LLM_agent/tests/test_tool_qrcode.py) —
семь юнит-тестов функций класса плюс общая функция `test_all_qrcode_unit_tests`,
которая вызывает их все подряд. Покрытие измеряется только для файла
`llm_agent/tool_qrcode.py` (см. `.github/workflows/qrcode-tool.yml`).

Локальный запуск:

```bash
cd 3_1_LLM_agent
pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests/test_tool_qrcode.py -v
```

Бэйджик покрытия обновляется через gist: в настройках репозитория нужны секреты
`GIST_SECRET` (токен со scope `gist`) и `GIST_ID` (идентификатор gist),
а в ссылке на бэйджик выше `GIST_ID` заменяется на этот же идентификатор.
Shields.io кеширует бэйджик ~5 минут.
