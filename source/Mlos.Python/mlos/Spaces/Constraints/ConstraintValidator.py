import ast

class ConstraintValidator(ast.NodeVisitor):
    """Validates constraints by walking their AST."""

    def visit(self, node: ast.AST) -> None:
        return super().visit(node)