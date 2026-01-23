import re
from typing import Tuple, Optional
import json
import requests
import logging
from gem.tools.search_tool import SearchTool as GEMSearchTool
from roll.utils.logging import get_logger

class XpengSearchTool(GEMSearchTool):

    def __init__(self, **kwargs):  
        super().__init__(**kwargs)  
        self.logger = get_logger()
        self.logger.info(f"[TOOL] XpengSearchTool initialized")  
        self.logger.info(f"[TOOL] search_url: {self.search_url}")  
        self.logger.info(f"[TOOL] topk: {self.topk}")

    def _search(self, query: str):
        """
        Perform a search using the configured search_url.
        Returns a formatted string of search results.
        """
        if not self._search_url_resolved:
            self.search_url = self.search_url or os.environ.get("SEARCH_URL")
            self._search_url_resolved = True

        if not self.search_url:
            raise ValueError("search_url must be provided for SearchTool.")

        payload = {"queries": [query], "topk": self.topk, "return_scores": True}
        self.logger.info(f"[TOOL] XpengSearchTool request paylaod:{payload}")  
        try:
            response = requests.post(
                self.search_url,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = json.loads(response.content)['result']
            self.logger.info(f"[TOOL] XpengSearchTool result {result}")  
            return self._passages2string(result)
        except Exception as e:
            return f"[SearchTool] Error: {e}]"
    
    def _passages2string(self, result):
        format_reference = ""
        for idx, doc_item in enumerate(result):
            content = doc_item['document']["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference