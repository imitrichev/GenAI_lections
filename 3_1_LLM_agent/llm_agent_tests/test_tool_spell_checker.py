from llm_agent.tools.tool_spell_checker import SpellCheckerTool


class TestEmptyAndWhitespace:
    def setup_method(self):
        self.tool = SpellCheckerTool()

    def test_empty_text_returns_message(self):
        assert self.tool.use("") == "Текст для проверки не предоставлен."

    def test_whitespace_only_returns_message(self):
        assert self.tool.use("   ") == "Текст для проверки не предоставлен."


class TestNoErrors:
    def test_no_errors_found(self):
        tool = SpellCheckerTool()
        result = tool.use("hello world")
        assert "Ошибок не найдено" in result


class TestCorrections:
    def test_corrections_found_english(self):
        tool = SpellCheckerTool()
        result = tool.use("helo world")
        assert "'helo'" in result
        assert "->" in result

    def test_corrections_found_russian(self):
        tool = SpellCheckerTool()
        result = tool.use("привет мр")
        assert "'мр'" in result
        assert "->" in result


class TestAttributes:
    def test_name_attribute(self):
        assert SpellCheckerTool.name == "spell_check"

    def test_description_attribute(self):
        assert SpellCheckerTool.description
        assert len(SpellCheckerTool.description) > 0
