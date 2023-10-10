from __future__ import annotations

import ast

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from .ContextAwareTranformer import ContextAwareTransformer

class ImportExtractor(ContextAwareTransformer):
    def __init__(self, path: Path) -> None:
        super().__init__(path=path)
        self.imports: List[ast.Import | ast.ImportFrom] = []

    def visit(self, node: ast.AST):
        return super().visit(node)

    def generic_visit(self, node: ast.AST) -> ast.AST:
        return super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> ast.AST:
        self.imports.append(node)
        return super().generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.Ast:
        self.imports.append(node)
        return super().generic_visit(node)


