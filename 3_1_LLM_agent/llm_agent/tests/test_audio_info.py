import os
import wave
import unittest

from llm_agent.tool_audio_info import AudioInfoTool


class TestAudioInfoTool(unittest.TestCase):

    def setUp(self):
        """Создаёт тестовый WAV-файл перед каждым тестом."""
        self.test_file = "test_audio.wav"

        with wave.open(self.test_file, "w") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(44100)

            # Создаём 1 секунду тишины
            audio.writeframes(b"\x00\x00" * 44100)

        self.tool = AudioInfoTool()

    def tearDown(self):
        """Удаляет тестовый файл после каждого теста."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_audio_file_exists(self):
        result = self.tool.use(self.test_file)

        self.assertIn("Информация об аудиофайле", result)
        self.assertIn("Длительность", result)

    def test_audio_file_not_found(self):
        result = self.tool.use("file_that_does_not_exist.wav")

        self.assertIn("не найден", result)

    def test_audio_parameters(self):
        result = self.tool.use(self.test_file)

        self.assertIn("Длительность: 1:00", result)
        self.assertIn("Количество каналов: 1 (моно)", result)
        self.assertIn("Частота дискретизации: 44100 Гц", result)


if __name__ == "__main__":
    unittest.main()
