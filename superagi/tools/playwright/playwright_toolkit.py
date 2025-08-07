from abc import ABC
from typing import List
from superagi.tools.base_tool import BaseToolkit, BaseTool
from superagi.tools.playwright.playwright import PlaywrightTool

class PlaywrightToolkit(BaseToolkit, ABC):
    name: str = "Playwright Toolkit"
    description: str = "A toolkit for interacting with web pages using Playwright."

    def get_tools(self) -> List[BaseTool]:
        return [PlaywrightTool()]

    def get_env_keys(self) -> List[str]:
        return []
