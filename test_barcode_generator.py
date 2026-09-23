import unittest
import os
import shutil
import tempfile
from barcode_generator import BarcodeGeneratorTool


class TestBarcodeGeneratorTool(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.tool = BarcodeGeneratorTool(output_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_generate_ean13_success(self):
        """Тест 1: Генерация EAN-13"""
        result = self.tool.generate_barcode("460123456789", format_type='ean13')
        self.assertTrue(os.path.exists(result['file_path']))
        self.assertEqual(result['format'], 'ean13')
        self.assertEqual(len(result['data']), 13)

    def test_generate_code128_success(self):
        """Тест 2: Генерация Code128"""
        result = self.tool.generate_barcode("TEST-123", format_type='code128')
        self.assertTrue(os.path.exists(result['file_path']))
        self.assertEqual(result['data'], "TEST-123")

    def test_invalid_ean13(self):
        """Тест 3: Ошибка при неверном EAN-13"""
        with self.assertRaises(ValueError):
            self.tool.generate_barcode("123", format_type='ean13')


if __name__ == '__main__':
    unittest.main()