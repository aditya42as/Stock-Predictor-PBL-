import time
import requests
from typing import List, Dict, Optional
from bs4 import BeautifulSoup

def fetch_headlines_newsapi(query: str, api_key: str, language: str = "en", page_size: int = 20) -> List[Dict]:
    if not api_key:
        raise ValueError("NewsAPI key required")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": language,
        "pageSize": page_size,
        "sortBy": "publishedAt",
    }
    headers = {"Authorization": api_key}
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for a in data.get("articles", []):
        results.append({
            "title": a.get("title"),
            "description": a.get("description"),
            "source": a.get("source", {}).get("name"),
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt")
        })
    return results


def fetch_headlines_google(query: str, max_items: int = 20, pause: float = 1.0) -> List[Dict]:
    q = requests.utils.requote_uri(query)
    url = f"https://news.google.com/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    items = []
    articles = soup.find_all("article")
    for a in articles:
        h = a.find("h3") or a.find("h4")
        if not h:
            continue
        title = h.get_text(strip=True)
        link_tag = h.find("a")
        href = link_tag["href"] if link_tag and link_tag.has_attr("href") else None
        url_full = None
        if href:
            if href.startswith("./"):
                url_full = "https://news.google.com" + href[1:]
            elif href.startswith("http"):
                url_full = href
        source_span = a.find("span", {"class": "xQ82C e8fRJf"})
        source = source_span.get_text(strip=True) if source_span else None
        snippet_tag = a.find("span", {"class": "xBbh9"})
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else None
        items.append({"title": title, "url": url_full, "source": source, "snippet": snippet})
        if len(items) >= max_items:
            break
    time.sleep(pause)
    return items


def fetch_headlines(query: str, api_key: Optional[str] = None, prefer_api: bool = True, max_items: int = 20) -> List[Dict]:
    if api_key and prefer_api:
        try:
            return fetch_headlines_newsapi(query, api_key=api_key, page_size=max_items)
        except:
            pass
    try:
        return fetch_headlines_google(query, max_items=max_items)
    except:
        return []
