import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.append(os.getcwd())

from llm.deepseek_client import resolve_publisher_url

class TestUrlResolver(unittest.TestCase):
    @patch('llm.deepseek_client.http')
    def test_direct_url(self, mock_http):
        url = "https://finance.yahoo.com/news/something.html"
        mock_resp = MagicMock()
        mock_resp.url = url
        mock_http.get.return_value = mock_resp
        
        result = resolve_publisher_url(url)
        self.assertEqual(result, url)

    @patch('llm.deepseek_client.http')
    def test_google_redirect_meta_refresh(self, mock_http):
        start_url = "https://news.google.com/rss/articles/123"
        target_url = "https://www.reuters.com/article/business"
        
        resp1 = MagicMock()
        resp1.url = start_url
        resp1.text = f'<html><head><meta http-equiv="refresh" content="0;url={target_url}"></head></html>'
        
        resp2 = MagicMock()
        resp2.url = target_url
        
        mock_http.get.side_effect = [resp1, resp2]
        
        result = resolve_publisher_url(start_url)
        self.assertEqual(result, target_url)

    @patch('llm.deepseek_client.http')
    def test_google_redirect_anchor_tag(self, mock_http):
        start_url = "https://news.google.com/rss/articles/456"
        target_url = "https://www.bloomberg.com/news/articles/2025-01-01/example"
        
        resp1 = MagicMock()
        resp1.url = start_url
        resp1.text = f'<html><body><c-wiz><a href="{target_url}" jsname="tljFtd">Open</a></c-wiz></body></html>'
        
        resp2 = MagicMock()
        resp2.url = target_url
        resp2.text = "<html>Content...</html>"
        
        mock_http.get.side_effect = [resp1, resp2]
        
        result = resolve_publisher_url(start_url)
        self.assertEqual(result, target_url)

    @patch('llm.deepseek_client.http')
    def test_avoid_favicon_links(self, mock_http):
        start_url = "https://news.google.com/rss/articles/789"
        bad_image_url = "https://lh3.googleusercontent.com/proxy/abc=w16-h16"
        good_url = "https://www.cnbc.com/2025/01/01/market-news.html"
        
        resp1 = MagicMock()
        resp1.url = start_url
        resp1.text = f'''
        <html>
            <img src="{bad_image_url}">
            <a href="{bad_image_url}">Icon</a>
            <a href="{good_url}">Read Full Article</a>
        </html>
        '''
        
        resp2 = MagicMock()
        resp2.url = good_url
        resp2.text = "<html>Real article...</html>"
        
        mock_http.get.side_effect = [resp1, resp2]
        
        result = resolve_publisher_url(start_url)
        self.assertEqual(result, good_url)

if __name__ == '__main__':
    unittest.main()
