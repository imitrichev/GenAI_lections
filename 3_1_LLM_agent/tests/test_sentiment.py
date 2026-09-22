from llm_agent.tool_sentiment import SentimentAnalyzerTool

def test_detects_positive_sentimennt():
    tool = SentimentAnalyzerTool()
    result = tool.use("Сегодня прекрасный день, я очень рад и всем доволен!")
    assert "позитивная" in result

def test_detects_negative_seentiment():
    tool = SentimentAnalyzerTool()
    result = tool.use("Это ужасно, я очень разочарован и расстроен.")
    assert "негативная" in result

def test_detects_neutral_sentiment():
    tool = SentimentAnalyzerTool()
    result = tool.use("Поезд отпраляется в десять часов утра со второго пути.")
    assert "нейтральная" in result

def test_empty_text():
    tool = SentimentAnalyzerTool()
    assert "Ошибка" in tool.use("")

def test_word_counts():
    tool = SentimentAnalyzerTool()
    result = tool.use("ОТЛИЧНО! Просто ЗАМЕЧАТЕЛЬНО и КРУТО.")
    normalized = " ".join(result.split())
    assert "позитивная" in normalized
    assert "Позитивных слов: 3" in normalized
    assert "Негативных слов: 0" in normalized