

import ast
import pytest
from pathlib import Path

from .ProjectCrawler import ProjectCrawler

class TestBuildBuilder:

    def test_crawler(self):
        crawler = ProjectCrawler(root=Path(__file__).parent.parent / "mlos")
        crawler.run()

        for path, file_imports in crawler.dependencies.items():
            print(path, len(file_imports.imports))
            for import_node in file_imports.imports:
                print(ast.dump(import_node))
            print("-----------------------------------------------------------------------------")
