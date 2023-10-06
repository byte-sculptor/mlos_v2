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

from collections import defaultdict, deque
from pathlib import Path
from typing import Iterator

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

    def run(self) -> None:
        num_files_by_extension: dict = defaultdict(lambda: 0)
        for file in self._iterate_file_paths():
            print(file)
            num_files_by_extension[file.suffix] += 1

        for extension, count in num_files_by_extension.items():
            print(f"{extension}: {count}")

    def _iterate_file_paths(self) -> Iterator[Path]:
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

