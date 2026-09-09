import unittest
import json
import os
from xml_parser_tool import XMLParserTool


class TestXMLParserTool(unittest.TestCase):
    """Юнит-тесты для класса XMLParserTool"""
    
    def test_parse_and_extract_by_xpath(self):
        """Тест 1: Парсинг XML и извлечение данных по XPath"""
        xml_data = """
        <library>
            <book id="1">
                <title>Python Programming</title>
                <author>John Doe</author>
            </book>
            <book id="2">
                <title>Data Science</title>
                <author>Jane Smith</author>
            </book>
        </library>
        """
        
        parser = XMLParserTool(xml_string=xml_data)
        titles = parser.extract_by_xpath(".//title")
        
        self.assertEqual(len(titles), 2)
        self.assertIn("Python Programming", titles)
        self.assertIn("Data Science", titles)
    
    def test_to_json_conversion(self):
        """Тест 2: Конвертация XML в JSON"""
        xml_data = """
        <person>
            <name>Alex</name>
            <age>25</age>
            <city>Moscow</city>
        </person>
        """
        
        parser = XMLParserTool(xml_string=xml_data)
        json_str = parser.to_json()
        json_data = json.loads(json_str)
        
        self.assertIn("person", json_data)
        self.assertEqual(json_data["person"]["name"], "Alex")
        self.assertEqual(json_data["person"]["age"], "25")
    
    def test_extract_with_attributes(self):
        """Тест 3: Извлечение элементов с атрибутами"""
        xml_data = """
        <catalog>
            <product id="101" category="electronics">
                <name>Laptop</name>
                <price>999</price>
            </product>
            <product id="102" category="books">
                <name>Python Book</name>
                <price>29</price>
            </product>
        </catalog>
        """
        
        parser = XMLParserTool(xml_string=xml_data)
        products = parser.extract_elements_by_xpath(".//product")
        
        self.assertEqual(len(products), 2)
        self.assertEqual(products[0].attrib["id"], "101")
        self.assertEqual(products[1].attrib["category"], "books")
    
    def test_save_json_to_file(self):
        """Тест 4: Сохранение JSON в файл (дополнительный)"""
        xml_data = """
        <config>
            <setting name="timeout">30</setting>
            <setting name="retries">3</setting>
        </config>
        """
        
        parser = XMLParserTool(xml_string=xml_data)
        output_file = "test_output.json"
        
        parser.save_json(output_file)
        
        # Проверяем, что файл создан
        self.assertTrue(os.path.exists(output_file))
        
        # Проверяем содержимое
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.assertIn("config", data)
        
        # Удаляем тестовый файл
        os.remove(output_file)


def run_all_tests():
    """Функция для запуска всех тестов"""
    unittest.main()


if __name__ == "__main__":
    run_all_tests()