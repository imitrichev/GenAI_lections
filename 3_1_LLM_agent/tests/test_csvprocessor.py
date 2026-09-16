from llm_agent.tool_csvprocessor import CSVProcessorTool


def test_read(tmp_path):
    csv_file = tmp_path / "data.csv"

    csv_file.write_text(
        "name,age,city\n"
        "Ivan,20,Moscow\n"
        "Anna,25,Kazan\n"
        "Petr,30,Moscow\n"
    )

    tool = CSVProcessorTool()

    result = tool.use(
        csv_file,
        operation="read"
    )

    assert "Ivan" in result


def test_filter(tmp_path):
    csv_file = tmp_path / "data.csv"

    csv_file.write_text(
        "name,age,city\n"
        "Ivan,20,Moscow\n"
        "Anna,25,Kazan\n"
        "Petr,30,Moscow\n"
    )

    tool = CSVProcessorTool()

    result = tool.use(
        csv_file,
        operation="filter",
        col="city",
        val="Moscow"
    )

    assert "Ivan" in result
    assert "Petr" in result
    assert "Anna" not in result


def test_stats(tmp_path):
    csv_file = tmp_path / "data.csv"

    csv_file.write_text(
        "name,age,city\n"
        "Ivan,20,Moscow\n"
        "Anna,25,Kazan\n"
        "Petr,30,Moscow\n"
    )

    tool = CSVProcessorTool()

    result = tool.use(
        csv_file,
        operation="stats",
        col="age"
    )

    assert "25.0" in result