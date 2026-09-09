import xml.etree.ElementTree as ET
import json
from typing import List, Dict, Any, Optional

class XMLParserTool:
    def __init__(self, xml_string: Optional[str] = None, xml_file: Optional[str] = None):
        """
        Инициализация парсера.
        
        :param xml_string: XML-строка для парсинга
        :param xml_file: Путь к XML-файлу для парсинга
        """
        self.tree = None
        self.root = None
        
        if xml_string:
            self.root = ET.fromstring(xml_string)
            self.tree = ET.ElementTree(self.root)
        elif xml_file:
            self.tree = ET.parse(xml_file)
            self.root = self.tree.getroot()
    
    def parse(self, xml_string: str = None, xml_file: str = None) -> ET.Element:
        """
        Парсинг XML из строки или файла.
        
        :param xml_string: XML-строка
        :param xml_file: Путь к файлу
        :return: Корневой элемент дерева
        """
        if xml_string:
            self.root = ET.fromstring(xml_string)
            self.tree = ET.ElementTree(self.root)
        elif xml_file:
            self.tree = ET.parse(xml_file)
            self.root = self.tree.getroot()
        
        return self.root
    
    def extract_by_xpath(self, xpath: str) -> List[str]:
        """
        Извлечение данных по XPath-запросу.
        
        :param xpath: XPath-выражение
        :return: Список найденных текстовых значений
        """
        if self.root is None:
            raise ValueError("XML не загружен. Вызовите parse() сначала.")
        
        # ElementTree поддерживает ограниченный XPath
        elements = self.root.findall(xpath)
        return [elem.text for elem in elements if elem.text]
    
    def extract_elements_by_xpath(self, xpath: str) -> List[ET.Element]:
        """
        Извлечение элементов по XPath (для дальнейшей обработки).
        
        :param xpath: XPath-выражение
        :return: Список найденных элементов
        """
        if self.root is None:
            raise ValueError("XML не загружен.")
        
        return self.root.findall(xpath)
    
    def to_dict(self, element: ET.Element = None) -> Dict[str, Any]:
        """
        Конвертация XML-элемента в словарь Python.
        
        :param element: Элемент для конвертации (по умолчанию корень)
        :return: Словарь
        """
        if element is None:
            element = self.root
        
        result = {element.tag: {} if element.attrib else None}
        children = list(element)
        
        if children:
            result[element.tag] = {child.tag: self.to_dict(child)[child.tag] for child in children}
        
        if element.attrib:
            result[element.tag].update(('@' + k, v) for k, v in element.attrib.items())
        
        if element.text:
            text = element.text.strip()
            if children or element.attrib:
                result[element.tag]['#text'] = text
            else:
                result[element.tag] = text
        
        return result
    
    def to_json(self, element: ET.Element = None, indent: int = 2) -> str:
        """
        Конвертация XML в JSON-строку.
        
        :param element: Элемент для конвертации (по умолчанию корень)
        :param indent: Отступ для форматирования
        :return: JSON-строка
        """
        data_dict = self.to_dict(element)
        return json.dumps(data_dict, indent=indent, ensure_ascii=False)
    
    def save_json(self, output_file: str, element: ET.Element = None):
        """
        Сохранение результата в JSON-файл.
        
        :param output_file: Путь к выходному файлу
        :param element: Элемент для конвертации
        """
        json_str = self.to_json(element)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_str)