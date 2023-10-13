from __future__ import annotations

import ast

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Union

from .ContextAwareTranformer import ContextAwareTransformer


@dataclass
class FileImports:
    path: Path
    imports: list[Union[ast.Import, ast.ImportFrom]] = field(default_factory=list)


class ImportExtractor(ContextAwareTransformer):
    def __init__(self, path: Path) -> None:
        super().__init__(path=path)
        self._imports: List[ast.Import | ast.ImportFrom] = []

    def get_imports(self) -> FileImports:
        return FileImports(
            path=self.path,
            imports=self._imports
        )

    def visit(self, node: ast.AST):
        return super().visit(node)

    def generic_visit(self, node: ast.AST) -> ast.AST:
        return super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> ast.AST:
        self._imports.append(node)
        return super().generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.Ast:
        self._imports.append(node)
        return super().generic_visit(node)
