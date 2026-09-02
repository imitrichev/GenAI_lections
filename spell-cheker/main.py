from pydantic import BaseModel
from spellchecker import SpellChecker


class ErrorResponse(BaseModel):
    raw_data: str
    suggestion: str | None


class SpellCheckerTool:

    def __init__(self):
        self.spellchecker = SpellChecker()

    def get_message_suggestions(self, message: str) -> list[ErrorResponse]:
        return [
            ErrorResponse(raw_data=word, suggestion=self.spellchecker.correction(word)) for word in message.split()
        ]

def main():
    user = input("Введите текст (на англ. или русском.): ")
    spell_tool = SpellCheckerTool()
    print(spell_tool.get_message_suggestions(user))


if __name__ == '__main__':
    main()
