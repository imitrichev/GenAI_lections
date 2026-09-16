# llm_agent/tool_imagetotext.py
import os
import tempfile
from urllib.parse import urlparse

import requests
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from pytesseract import TesseractNotFoundError


class ImageToTextTool:
    """Инструмент для извлечения текста из изображений с помощью OCR (Tesseract)."""

    name = "image_to_text"
    description = (
        "Извлекает текст из изображений с помощью OCR (Tesseract). "
        "Принимает локальный путь к изображению или URL. "
        "Поддерживает форматы: JPG, JPEG, PNG, BMP, TIFF."
    )

    def use(self, source: str, lang: str = "rus+eng", preprocess: bool = True) -> str:
        """
        Извлекает текст из изображения с помощью OCR.

        Args:
            source: путь к локальному изображению или URL
            lang: языки для распознавания (по умолчанию "rus+eng")
            preprocess: применять предобработку изображения (улучшает качество OCR)

        Returns:
            Строка с извлеченным текстом или сообщение об ошибке.
        """
        if not source or not source.strip():
            return "Ошибка: пустой источник (source). Укажите путь к файлу или URL."

        temp_file = None
        try:
            print(f"> Обрабатываю изображение: '{source}'")

            is_url = urlparse(source).scheme in ("http", "https")

            if is_url:
                print("> Скачиваю изображение по URL...")
                response = requests.get(source, timeout=30, stream=True)
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "")
                if "image" not in content_type.lower():
                    return (
                        f"Ошибка: по URL '{source}' не найдено изображение "
                        f"(Content-Type: {content_type})"
                    )

                ext = self._get_extension_from_url(source)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file.close()
                image_path = temp_file.name
                print(f"> Изображение сохранено во временный файл: {image_path}")
            else:
                image_path = source
                if not os.path.isfile(image_path):
                    return f"Ошибка: файл '{image_path}' не найден."

            # Проверяем, что файл действительно изображение
            try:
                img = Image.open(image_path)
                print(f"> Размер изображения: {img.size}, формат: {img.format}")
            except Exception as e:
                return f"Ошибка: файл не является корректным изображением. {e}"

            if preprocess:
                img = self._preprocess_image(img)

            print(f"> Запускаю OCR (языки: {lang})...")
            extracted_text = pytesseract.image_to_string(img, lang=lang)

            cleaned_text = extracted_text.strip()

            if not cleaned_text:
                return "(текст не найден на изображении)"

            result = (
                f"Извлеченный текст из: {source}\n"
                f"{'=' * 50}\n"
                f"{cleaned_text}\n"
                f"{'=' * 50}\n"
                f"Длина текста: {len(cleaned_text)} символов"
            )

            print(f"> Успешно извлечено {len(cleaned_text)} символов")
            return result

        except TesseractNotFoundError:
            return (
                "Ошибка: Tesseract не установлен. "
                "Установите Tesseract OCR (https://github.com/tesseract-ocr/tesseract)"
            )
        except Exception as e:
            print(f"> Ошибка при обработке изображения: {e}")
            return f"Произошла ошибка при извлечении текста из изображения '{source}': {e}"

        finally:
            if temp_file is not None:
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass

    def _preprocess_image(self, image):
        """Предобработка изображения для улучшения качества OCR."""
        if image.mode != 'L':
            image = image.convert('L')

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)

        image = image.filter(ImageFilter.SHARPEN)

        return image

    def _get_extension_from_url(self, url: str) -> str:
        """Определяет расширение файла по URL."""
        path = urlparse(url).path
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            return ext
        return '.jpg'


# Пример использования:
# tool = ImageToTextTool()
# print(tool.use("https://example.com/image.jpg"))
# print(tool.use("/path/to/local/image.png", lang="eng"))