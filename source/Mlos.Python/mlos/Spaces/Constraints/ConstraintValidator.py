import ast

from mlos.Exceptions import InvalidConstraintException

class ConstraintValidator(ast.NodeVisitor):
    """Validates constraints by walking their AST."""

    def __init__(self, allowed_variable_names: list[str]) -> None:
        # List of variables present in the AST
        self.variable_names: list[str] = []

        self._validated_root: bool = False

        self._allowed_variable_names: set[str] = set(allowed_variable_names)

    def visit_Expression(self, node: ast.AST) -> None:
        if not self._validated_root:
            self._validate_root(node=node)
            self._validated_root = True
        return self.generic_visit(node)

    def _validate_root(self, node: ast.AST) -> None:
        if not isinstance(node, ast.Expression):
            raise InvalidConstraintException("Constraint must be a single expression.")
        if not isinstance(node.body, ast.Compare):
            raise InvalidConstraintException("Constraint must be a single comparsion operation.")

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            if not isinstance(op, (ast.Lt, ast.LtE, ast.Eq, ast.NotEq, ast.Gt, ast.GtE)):
                raise InvalidConstraintException(f"Invalid comparison: {op=}")

        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load): raise InvalidConstraintException()
        if not node.id in self._allowed_variable_names:
            raise InvalidConstraintException(f"{node.id=}, {self._allowed_variable_names=}")
        self.variable_names.append(node.id)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        allowed_function_names = set(["abs", "sum"])

        if not isinstance(node.func, ast.Name):
            raise InvalidConstraintException()
        if not node.func.id in allowed_function_names:
            raise InvalidConstraintException(f"Disallowed function call: {node.func.id} at {node.lineno} {node.col_offset}")

        for arg in node.args:
            self.generic_visit(arg)
        for keyword in node.keywords:
            self.generic_visit(keyword)
        return None


