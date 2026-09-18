import ast
import logging
import operator

from langchain.tools import BaseTool, ToolException

logger = logging.getLogger(__name__)

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_MAX_POWER_EXPONENT = 1000


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        if isinstance(node.op, ast.Pow):
            exponent = _eval_node(node.right)
            if abs(exponent) > _MAX_POWER_EXPONENT:
                raise ToolException("Exponent too large.")
        return _OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))

    raise ToolException("Expression contains unsupported syntax.")


def safe_eval_arithmetic(expression: str) -> float:
    """Evaluate a simple arithmetic expression without using eval()."""

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolException(f"Could not parse expression: {expression}") from exc

    return _eval_node(tree.body)


class CalculatorTool(BaseTool):
    """A simple calculator implemented directly with BaseTool."""

    name: str = "calculator"
    description: str = (
        "Performs arithmetic calculations (+, -, *, /, **). "
        "Use this tool when the user asks for a mathematical calculation."
    )

    def _run(self, expression: str) -> str:
        try:
            result = safe_eval_arithmetic(expression)
            logger.info("calculator tool called with %r", expression)
            return str(result)
        except ToolException:
            raise
        except Exception as exc:
            raise ToolException(f"Could not calculate expression: {expression}") from exc


calculator_tool = CalculatorTool(handle_tool_error=True)
