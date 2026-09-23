import unittest
import tempfile
import os
from PIL import Image
import qrcode
from llm_agent.tool_qrdecoder import QRDecoderTool


class TestQRDecoderTool(unittest.TestCase):

    def setUp(self):
        """Подготовка тестовых данных"""
        self.tool = QRDecoderTool()
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = "Hello, World!"
        self.qr_filename = os.path.join(self.temp_dir, "test_qr.png")
        img = qrcode.make(self.test_data)
        img.save(self.qr_filename)

    def tearDown(self):
        if os.path.exists(self.qr_filename):
            os.remove(self.qr_filename)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_decode_local_file(self):
        """Тест 1: Декодирование локального файла"""
        result = self.tool.use(self.qr_filename)
        self.assertIn(self.test_data, result)
        self.assertIn("успешно декодирован", result)

    def test_decode_invalid_file(self):
        """Тест 2: Обработка несуществующего файла"""
        result = self.tool.use("non_existent_file.png")
        self.assertIn("файл не найден", result)

    def test_decode_image_without_qr(self):
        """Тест 3: Изображение без QR-кода"""
        empty_img_path = os.path.join(self.temp_dir, "empty.png")
        Image.new('RGB', (100, 100), color='white').save(empty_img_path)
        result = self.tool.use(empty_img_path)
        # Проверяем, что либо QR не найден, либо вернулась ошибка
        self.assertTrue(
            "QR-код не найден" in result or "Ошибка" in result,
            f"Неожиданный результат: {result}"
        )
        os.remove(empty_img_path)


if __name__ == '__main__':
    unittest.main()