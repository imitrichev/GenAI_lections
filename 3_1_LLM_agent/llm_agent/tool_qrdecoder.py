from pyzbar.pyzbar import decode
from PIL import Image
import requests
from io import BytesIO
import os


class QRDecoderTool:
    """
    Инструмент для декодирования QR-кодов из изображений.
    Поддерживает локальные файлы и URL.
    """

    def __init__(self):
        self.name = "qr_decoder"
        self.description = "Декодирует QR-код из изображения по пути или URL"

    def use(self, input_data: str) -> str:
        """
        Декодирует QR-код из изображения.

        Args:
            input_data (str): Путь к локальному файлу или URL изображения

        Returns:
            str: Декодированное содержимое QR-кода или сообщение об ошибке
        """
        try:
            # Загружаем изображение
            if input_data.startswith(('http://', 'https://')):
                response = requests.get(input_data, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                if not os.path.exists(input_data):
                    return f"Ошибка: файл не найден по пути {input_data}"
                image = Image.open(input_data)

            # Декодируем QR-код
            decoded_objects = decode(image)

            if not decoded_objects:
                return "QR-код не найден на изображении"

            # Извлекаем данные из первого найденного QR-кода
            result = decoded_objects[0].data.decode('utf-8')
            return f"QR-код успешно декодирован: {result}"

        except Exception as e:
            return f"Ошибка при декодировании QR-кода: {str(e)}"