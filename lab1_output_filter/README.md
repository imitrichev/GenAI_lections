# Lab 1 — OutputFilter (вариант 12)

![Coverage](coverage.svg)

Класс `OutputFilter` проверяет текстовый ответ LLM на наличие
конфиденциальной информации (email-адреса, пароли, API-ключи) с помощью
регулярных выражений и умеет заменять найденные фрагменты маской вида
`[REDACTED:category]`.

## Использование

```python
from output_filter import OutputFilter

f = OutputFilter()
text = "Мой email ivan@example.com, а пароль password: hunter2"

f.contains_sensitive(text)  # True
f.redact(text)              # "Мой email [REDACTED:email], а пароль [REDACTED:password]"
```

## Запуск тестов

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest tests -v
```
