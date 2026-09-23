from barcode_generator import BarcodeGeneratorTool


def main():
    print("=" * 50)
    print("ГЕНЕРАТОР ШТРИХ-КОДОВ")
    print("=" * 50)

    # Создаем инструмент
    tool = BarcodeGeneratorTool(output_dir="barcodes")

    # Тестовые данные
    test_data = [
        ("460123456789", "EAN-13"),
        ("TEST-CODE-128", "Code128"),
    ]

    for data, format_type in test_data:
        print(f"\nГенерация {format_type}...")
        print(f" Данные: {data}")

        try:
            result = tool.generate_barcode(data, format_type=format_type.lower())
            print(f"   Создан: {result['file_path']}")
            print(f"   Данные: {result['data']}")
        except Exception as e:
            print(f"  Ошибка: {e}")

    print("\n" + "=" * 50)
    print("Генерация завершена!")
    print("Результаты в папке 'barcodes/'")
    print("=" * 50)


if __name__ == "__main__":
    main()