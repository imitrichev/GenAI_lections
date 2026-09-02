from mutagen import File


class AudioInfoTool:
    """Инструмент для извлечения метаданных аудиофайла."""

    name = "audio_info"
    description = (
        "Извлекает метаданные аудиофайла: длительность, битрейт, "
        "количество каналов, частоту дискретизации и другие параметры. "
        "Поддерживает MP3, WAV и другие аудиоформаты."
    )

    def use(self, file_path: str) -> str:
        """
        Извлекает и возвращает информацию об аудиофайле.
        """
        try:
            print(f"> Получаю информацию об аудиофайле: '{file_path}'")

            audio = File(file_path)

            if audio is None:
                return (
                    f"Не удалось определить формат аудиофайла "
                    f"'{file_path}'."
                )

            info = audio.info

            # Длительность в секундах
            duration = getattr(info, "length", None)

            if duration is not None:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                duration_str = f"{minutes}:{seconds:02d}"
            else:
                duration_str = "Неизвестно"

            # Битрейт
            bitrate = getattr(info, "bitrate", None)

            if bitrate is not None:
                bitrate_str = f"{bitrate // 1000} кбит/с"
            else:
                bitrate_str = "Неизвестно"

            # Количество каналов
            channels = getattr(info, "channels", None)

            if channels is not None:
                if channels == 1:
                    channels_str = "1 (моно)"
                elif channels == 2:
                    channels_str = "2 (стерео)"
                else:
                    channels_str = str(channels)
            else:
                channels_str = "Неизвестно"

            # Частота дискретизации
            sample_rate = getattr(info, "sample_rate", None)

            if sample_rate is not None:
                sample_rate_str = f"{sample_rate} Гц"
            else:
                sample_rate_str = "Неизвестно"

            # Формат
            format_name = audio.mime[0] if audio.mime else "Неизвестно"

            result = (
                f"Информация об аудиофайле:\n\n"
                f"Файл: {file_path}\n"
                f"Формат: {format_name}\n"
                f"Длительность: {duration_str}\n"
                f"Битрейт: {bitrate_str}\n"
                f"Количество каналов: {channels_str}\n"
                f"Частота дискретизации: {sample_rate_str}"
            )

            # Дополнительные метаданные
            if audio.tags:
                result += "\n\nМетаданные:"

                for key, value in audio.tags.items():
                    result += f"\n{key}: {value}"

            print("> Информация об аудиофайле успешно получена.")

            return result

        except FileNotFoundError:
            print(f"> Файл не найден: {file_path}")
            return f"Файл '{file_path}' не найден."

        except Exception as e:
            print(f"> Ошибка при обработке аудиофайла: {e}")
            return (
                f"Произошла ошибка при обработке "
                f"аудиофайла '{file_path}': {e}"
            )