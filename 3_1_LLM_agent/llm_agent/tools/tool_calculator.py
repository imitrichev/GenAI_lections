# llm_agent/tool_calculator.py

import ast
import operator as op

ALLOWED_OPERATORS = {
    ast.Add: op.add,    # Сложение (+)
    ast.Sub: op.sub,    # Вычитание (-)
    ast.Mult: op.mul,   # Умножение (*)
    ast.Div: op.truediv, # Деление (/)
    ast.Pow: op.pow,    # Возведение в степень (**)
    ast.USub: op.neg,   # Унарный минус (для отрицательных чисел)
}

class CalculatorTool:
    """Инструмент для безопасного выполнения математических выражений."""
    
    name = "calculator"
    description = "Выполняет арифметические вычисления. Пример: (5 + 3) * 2. Поддерживает: +, -, *, /, **"

    def use(self, expression: str) -> str:
        """
        Принимает строку с математическим выражением и возвращает результат.
        
        Args:
            expression (str): Выражение для вычисления, например, "10 * (2 + 3)".
            
        Returns:
            str: Строка с результатом или сообщением об ошибке.
        """
        try:
            node = ast.parse(expression, mode='eval').body
            result = self._eval_ast_node(node)
            return f"Результат вычисления '{expression}': {result}"
        except (ValueError, SyntaxError, TypeError) as e:
            return f"Ошибка: не могу вычислить выражение '{expression}'. Проверьте синтаксис. Детали: {e}"

    def _eval_ast_node(self, node):
        """Рекурсивно вычисляет узел AST."""
        if isinstance(node, ast.Constant): # В Python 3.8+ рекомендуется использовать ast.Constant вместо ast.Num
            return node.value
        elif isinstance(node, ast.BinOp): # Бинарная операция (например, 2 + 3)
            return ALLOWED_OPERATORS[type(node.op)](
                self._eval_ast_node(node.left),
                self._eval_ast_node(node.right)
            )
        elif isinstance(node, ast.UnaryOp): # Унарная операция (например, -5)
            return ALLOWED_OPERATORS[type(node.op)](self._eval_ast_node(node.operand))
        else:
            raise TypeError(f"Операция {node} не поддерживается")
