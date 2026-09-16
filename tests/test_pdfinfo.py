"""
Юнит-тесты для PDFInfoTool
"""

import os
import tempfile
import pytest
from pypdf import PdfWriter

from llm_agent.tool_pdfinfo import PDFInfoTool


@pytest.fixture
def simple_pdf_path():
    """Создаёт PDF из 3 страниц с метаданными."""
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=200, height=200)
    writer.add_metadata({
        "/Author": "Test Author",
        "/Title": "Test Document",
    })
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    writer.write(tmp.name)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


def test_page_count(simple_pdf_path):
    """get_page_count возвращает корректное число страниц."""
    tool = PDFInfoTool()
    assert tool.get_page_count(simple_pdf_path) == 3


def test_metadata_extraction(simple_pdf_path):
    """get_metadata возвращает автора и заголовок."""
    tool = PDFInfoTool()
    meta = tool.get_metadata(simple_pdf_path)
    assert meta.get("/Author") == "Test Author"
    assert meta.get("/Title") == "Test Document"


def test_extract_text_blank_pages(simple_pdf_path):
    """На пустых страницах текст пустой, но метод не падает."""
    tool = PDFInfoTool()
    text = tool.extract_text(simple_pdf_path)
    assert isinstance(text, str)


def test_extract_text_specific_page_out_of_range(simple_pdf_path):
    """Запрос несуществующей страницы → IndexError."""
    tool = PDFInfoTool()
    with pytest.raises(IndexError):
        tool.extract_text(simple_pdf_path, page=99)


def test_file_not_found():
    """Несуществующий файл → FileNotFoundError."""
    tool = PDFInfoTool()
    with pytest.raises(FileNotFoundError):
        tool.get_page_count("nope_does_not_exist.pdf")


def test_use_returns_summary(simple_pdf_path):
    """use() возвращает строку с ключевыми полями."""
    tool = PDFInfoTool()
    result = tool.use(simple_pdf_path)
    assert "Страниц: 3" in result
    assert "Test Author" in result