from __future__ import annotations

import ast


from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class NodeContextStack:
    """Stack of NodeContext objects.

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
    def __init__(self, file_path: Path) -> None:
        self.file_path: Path = file_path
        self.stack: deque[NodeContext] = deque()

    def push(self, node_context: NodeContext) -> None:
        self.stack.append(node_context)

    def pop(self) -> NodeContext:
        assert len(self.stack) > 0
        return self.stack.pop()

    def peek(self) -> NodeContext:
        assert len(self.stack) > 0
        return self.stack[-1]

    def __len__(self) -> int:
        return len(self.stack)

@dataclass
class NodeContext:
    node: ast.AST
    context: Any = None # probably a dict? Or maybe some other small class?

class ContextAwareTransformer(ast.NodeTransformer):
    def __init__(
        self,
        path: Path
    ) -> None:
        self.path: Path = path
        self.node_context_stack: NodeContextStack = NodeContextStack(file_path=path)

    def visit(self, node: ast.AST):
        # TODO: would be good to somehow save the index into the parent's array or the name of the attribute
        # TODO: on the parent's Node. This would allow for unambiguous XPath addressing for each node.
        # TODO: the way to do this, would be to have specialized `visit_{NodeType}` methods that know how
        # TODO: to add this piece of information to their children.
        # TODO: or maybe make use of `ast.iter_fields` and `ast.iter_child_nodes`.
        self.node_context_stack.push(NodeContext(node=node))
        try:
            return super().visit(node)
        finally:
            self.node_context_stack.pop()
