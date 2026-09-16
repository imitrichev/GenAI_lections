
"""Агрегированный прогон всех тестов"""
import pytest
import sys


def run_all_suites():
    """Вызывает все тесты из tests/ внутри одной функции."""
    exit_code = pytest.main(["tests/", "-v", "--tb=short"])
    return exit_code


if __name__ == "__main__":
    sys.exit(run_all_suites())