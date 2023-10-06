

import pytest

from pathlib import Path

import os
import sys

from .ProjectCrawler import ProjectCrawler

class TestBuildBuilder:

    def test_crawler(self):
        crawler = ProjectCrawler(root=Path(__file__).parent.parent / "mlos")
        crawler.run()
