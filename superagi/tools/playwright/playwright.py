from typing import Type, Any
from pydantic import BaseModel, Field
from playwright.sync_api import sync_playwright
from superagi.tools.base_tool import BaseTool

class PlaywrightInput(BaseModel):
    url: str = Field(..., description="The URL to navigate to.")

class PlaywrightTool(BaseTool):
    name: str = "Playwright"
    args_schema: Type[BaseModel] = PlaywrightInput
    description: str = "A tool to interact with web pages using Playwright."

    def _execute(self, url: str):
        """
        Execute the Playwright tool.

        Args:
            url (str): The URL to navigate to.

        Returns:
            str: The page content.
        """
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(url)
                content = page.content()
                browser.close()
                return content
        except Exception as e:
            return f"Error: {e}"
