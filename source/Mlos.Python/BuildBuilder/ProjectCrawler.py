""" Builds the build files.

The basic idea here is to:
 * walk the project files
 * for each file walk the AST
 * find all the imports
 * build a DAG
 * express the DAG as set of bazel build files
 * write them

It's only temporarily living in this repo - will move it to a separate repo when it's done
so that it can be used elsewhere too.

"""
import ast
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Union


from .ImportExtractor import ImportExtractor, FileImports


class ProjectCrawler:
    """Crawls a project directory structure.

    This will be really simple. The crawler needs to do just the following:
    1. Start in some root directory (maybe multiple? would that be helpful at all?)
    2. Recursively iterate over files and directories. And:
        1. For each file, invoke whatever callbacks we have for that file type (or file selector? do we need to be that general?)
        2. For each directory add it to a stack to walk it later.
        3. Maintain a `realpath` cache. No need to walk the same files/dirs multiple times if for some reason someone decided to
            symlink files in their repo. UPDATE: don't, just assert that path == realpath and support symlinks some other day.

        For now this can all happen serially, or on multiple threads (given the I/O required), but ultimately we'd want this to run
    on multiple cores and be really blazing fast. Who knows - maybe rewrite some of it in rust.

    Architectural decisions:
        1. Who owns the DAG: project crawler
        2. What do file parsers return: a bunch of DAG edges to be added to the DAG. Nice place to check for cycles too :).
    """

    def __init__(self, root: Path) -> None:
        self.root: Path = root
        self.dependencies: dict[Path, FileImports] = {}

    def run(self) -> None:
        for file_path in self._iterate_file_paths():
            if file_path.suffix == ".py":
                self._visit_file(file_path=file_path)

    def _iterate_file_paths(self) -> Iterator[Path]:
        """Recursively iterates all file paths in self.root"""
        dir_stack: deque[Path] = deque()
        dir_stack.append(self.root)
        while len(dir_stack) > 0:
            current_dir: Path = dir_stack.pop()
            assert current_dir.is_dir()
            assert current_dir.exists()
            for entry in current_dir.iterdir():
                if entry.is_dir():
                    dir_stack.append(entry)
                elif entry.is_file():
                    yield entry
                else:
                    assert False, f"{entry=} is neither a file, nor a dir"

    def _visit_file(self, file_path: Path) -> None:
        #print(f"Visiting: {file_path}")
        with open(file_path, "r", encoding="utf-8") as in_file:
            source_str: str = in_file.read()

        tree: ast.AST = ast.parse(source_str)
        import_extractor: ImportExtractor = ImportExtractor(path=file_path)

        # The idea is that import extractor would accumulate imports and we can read them later.
        import_extractor.visit(tree)

        self.dependencies[file_path] = import_extractor.get_imports()

        for import_node in import_extractor._imports:
            #print(ast.dump(import_node))
            if isinstance(import_node, ast.Import):
                for alias in import_node.names:
                    dot_separated_module_path = alias.name
                    #print(f"{dot_separated_module_path=}")
            elif isinstance(import_node, ast.ImportFrom):
                dot_separated_module_path = import_node.module
                #print(f"{dot_separated_module_path=}")

            else:
                assert False, f"{type(import_node)}, {ast.dump(import_node)}"


        #print("--------------------------------------------------")

