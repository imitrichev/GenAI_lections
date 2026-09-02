from pydantic import BaseModel
from spellchecker import SpellChecker


class ErrorResponse(BaseModel):
    suggestion: str


class SpellCheckerTool:

    def __init__(self):
        self.spellchecker = SpellChecker()

    def get_message_suggestions(self, message: str) -> list[ErrorResponse]:
        pass


def main():
    user = input()
    spell_tool = SpellCheckerTool()
    print(spell_tool.get_message_suggestions(user))


if __name__ == '__main__':
    main()
