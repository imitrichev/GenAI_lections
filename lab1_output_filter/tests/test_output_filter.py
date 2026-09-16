from output_filter import OutputFilter


def test_redacts_email_address():
    output_filter = OutputFilter()
    text = "Свяжитесь со мной по адресу ivan.petrov@example.com, я отвечу."

    redacted = output_filter.redact(text)

    assert "ivan.petrov@example.com" not in redacted
    assert "[REDACTED:email]" in redacted


def test_redacts_password_and_api_key():
    output_filter = OutputFilter()
    text = "password: hunter2, а ключ api_key: sk-AbCdEfGh0123456789XYZ"

    redacted = output_filter.redact(text)

    assert "hunter2" not in redacted
    assert "sk-AbCdEfGh0123456789XYZ" not in redacted
    assert "[REDACTED:password]" in redacted
    assert "[REDACTED:api_key]" in redacted


def test_clean_text_is_left_untouched():
    output_filter = OutputFilter()
    text = "Погода в Москве сегодня солнечная, без осадков."

    assert output_filter.contains_sensitive(text) is False
    assert output_filter.redact(text) == text


def test_scan_returns_findings_with_correct_positions():
    output_filter = OutputFilter()
    text = "email: test@mail.ru"

    findings = output_filter.scan(text)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.category == "email"
    assert text[finding.start:finding.end] == "test@mail.ru"


def test_aws_style_api_key_is_detected():
    output_filter = OutputFilter()
    text = "AWS access key: AKIAIOSFODNN7EXAMPLE в логах"

    assert output_filter.contains_sensitive(text) is True


def run_all_output_filter_tests():
    """Отдельная функция, вызывающая все юнит-тесты класса OutputFilter подряд."""
    test_redacts_email_address()
    test_redacts_password_and_api_key()
    test_clean_text_is_left_untouched()
    test_scan_returns_findings_with_correct_positions()
    test_aws_style_api_key_is_detected()
    print("Все тесты OutputFilter прошли успешно.")


if __name__ == "__main__":
    run_all_output_filter_tests()
