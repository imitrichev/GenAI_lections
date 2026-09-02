# llm_agent/tool_excelreader.py

import pandas as pd


class ExcelReaderTool:
    """Инструмент для чтения данных из Excel-файлов."""

    name = "excel_reader"
    description = (
        "Читает данные из Excel-файла (по локальному пути или URL) "
        "и возвращает первые несколько строк таблицы в виде текста."
    )

    def use(self, file_path_or_url: str, rows_limit: int = 5) -> str:
        """Читает Excel-файл и возвращает его содержимое в текстовом формате.

        :param file_path_or_url: Локальный путь к файлу или URL.
        :param rows_limit: Количество первых строк для отображения (по
        умолчанию 5).
        """
        try:
            print(f"> Читаю Excel-файл: '{file_path_or_url}'")

            # Pandas использует 'openpyxl' под капотом для файлов .xlsx
            df = pd.read_excel(file_path_or_url)

            if df.empty:
                return f"Файл '{file_path_or_url}' не содержит данных."

            total_rows, total_cols = df.shape
            print(f"> Файл успешно прочитан. Всего строк: {total_rows}, колонок: {total_cols}.")

            # Берём первые N строк
            preview_df = df.head(rows_limit)

            # Преобразуем таблицу в текстовое представление
            table_text = preview_df.to_string(index=False)

            result = (
                f"Данные из файла (первые {len(preview_df)} из {total_rows} строк, всего колонок: {total_cols}):\n\n"
                f"{table_text}"
            )
            return result

        except Exception as e:
            print(f"> Ошибка при чтении Excel-файла: {e}")
            return f"Произошла ошибка при чтении Excel-файла '{file_path_or_url}': {e}"