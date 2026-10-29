"""Safe arithmetic evaluator. Parses with `ast` and walks an allowlist of nodes; never calls eval()."""

import ast
import math
import operator

from .base import ToolResult, error

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e}

MAX_EXPR_CHARS = 500
MAX_EXPONENT = 1000


class CalcError(ValueError):
    pass


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise CalcError(f"unknown name '{node.id}'")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalcError(f"exponent larger than {MAX_EXPONENT}")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        if node.keywords:
            raise CalcError("keyword arguments are not supported")
        return _FUNCS[node.func.id](*(_eval(arg) for arg in node.args))
    raise CalcError(f"unsupported syntax: {ast.dump(node)[:60]}")


def _format(value: float) -> str:
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def calculator(expression: str) -> ToolResult:
    expression = expression.strip()
    if not expression:
        return error("empty expression")
    if len(expression) > MAX_EXPR_CHARS:
        return error(f"expression longer than {MAX_EXPR_CHARS} characters")
    if "," in expression and "(" not in expression:
        return error("remove thousands separators (commas) from numbers, e.g. 1234567 not 1,234,567")
    try:
        tree = ast.parse(expression, mode="eval")
        return ToolResult(ok=True, content=_format(_eval(tree)))
    except ZeroDivisionError:
        return error("division by zero")
    except (CalcError, SyntaxError, TypeError, ValueError, OverflowError) as exc:
        return error(f"could not evaluate '{expression}': {exc}")
