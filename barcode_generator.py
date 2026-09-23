import barcode
from barcode.writer import ImageWriter
import os
from datetime import datetime
import hashlib


class BarcodeGeneratorTool:
    """
    Класс для генерации штрих-кодов
    """

    SUPPORTED_FORMATS = {
        'ean13': barcode.get_barcode_class('ean13'),
        'code128': barcode.get_barcode_class('code128'),
    }

    def __init__(self, output_dir="barcodes", enable_cache=True):
        self.output_dir = output_dir
        self.enable_cache = enable_cache
        self.cache = {}

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_barcode(self, data, format_type='ean13'):
        """Генерация штрих-кода"""
        if format_type not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format_type}")

        if format_type == 'ean13':
            data = self._validate_ean13(data)

        cache_key = self._get_cache_key(data, format_type)
        if self.enable_cache and cache_key in self.cache:
            return self.cache[cache_key]

        barcode_class = self.SUPPORTED_FORMATS[format_type]
        writer = ImageWriter()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"barcode_{format_type}_{timestamp}"

        barcode_obj = barcode_class(data, writer=writer)
        full_path = os.path.join(self.output_dir, filename)
        barcode_obj.save(full_path)
        file_path = f"{full_path}.png"

        result = {
            'data': data,
            'format': format_type,
            'file_path': file_path,
            'filename': filename,
        }

        if self.enable_cache:
            self.cache[cache_key] = result

        return result

    def _validate_ean13(self, data):
        """Валидация EAN-13"""
        data = ''.join(filter(str.isdigit, str(data)))

        if len(data) == 12:
            check_digit = self._calculate_ean13_checkdigit(data)
            return data + str(check_digit)
        elif len(data) == 13:
            return data
        else:
            raise ValueError(f"EAN-13 must be 12 or 13 digits, got: {len(data)}")

    def _calculate_ean13_checkdigit(self, data):
        """Расчет контрольной суммы EAN-13"""
        if len(data) != 12:
            raise ValueError("Need exactly 12 digits")

        odd_sum = sum(int(data[i]) for i in range(0, 12, 2))
        even_sum = sum(int(data[i]) for i in range(1, 12, 2))
        total = odd_sum + even_sum * 3
        check_digit = (10 - (total % 10)) % 10
        return check_digit

    def _get_cache_key(self, data, format_type):
        """Ключ для кеша"""
        key_string = f"{data}_{format_type}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def clear_cache(self):
        """Очистка кеша"""
        self.cache.clear()