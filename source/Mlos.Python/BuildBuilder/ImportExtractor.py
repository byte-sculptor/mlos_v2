

import ast


from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class NodeContext:
    """Holds context for an AST Node.

    When a context aware visitor walks the AST, it puts a NodeContext object
    on the stack, before visiting any of its children. Children have access
    to this stack. This basically allows descendant nodes to view their ancestry,
    as well as mutate their ancestors' contexts. Thus, an ancestor can transform
    itself in response to how its descendants mutated its context.

    Here's an example - it's currently impossible to wrap arbitrary `await` statements and
    expresions inside a context manager (e.g. to time it). Standalone `await` statements`
    can in theory transform themselves, but it becomes impossible if they are used inside
    an if statement or as part of a larger expression. Nor is it practical for every
    node to check if any of its descendants are await expressions, because the vast majority
    of the time they are not.

    With the NodeContext stack, an await expression node can simply notify its ancestors that
    it exists (by modifying the ancestral context) and thus the ancestral node transformers
    can inject tracing or rewrite code appropriately.
    """
    node: ast.AST
    context: Any = None# probably a dict? Or maybe some other small class?
class ContextAwareVisitor(ast.NodeTransformer):


    def __init__(
        self,
        path: Path
    ) -> None:
        self.path: Path = path
        self.node_context_stack: deque[NodeContext] = deque()

    def visit(self, node: ast.AST):
        return super().visit(node)

    def generic_visit(self, node: ast.AST) -> ast.AST:
        return super().generic_visit(node)


