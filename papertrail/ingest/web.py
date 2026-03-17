import logging
from typing import Dict, Tuple

from papertrail.utils.text import normalize_url

logger = logging.getLogger(__name__)

try:
    import requests
    from bs4 import BeautifulSoup
    _SUPPORT = True
except ImportError:
    _SUPPORT = False

_UA    = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
_NOISE = ["script", "style", "nav", "footer", "header", "aside", "noscript", "form", "button", "iframe"]


def load_url(url: str) -> Tuple[str, Dict[int, str], str]:
    """Return (text, section_map, err)."""
    if not _SUPPORT:
        return "", {}, "requests/beautifulsoup4 not installed."
    url = normalize_url(url)
    if not url:
        return "", {}, "Please enter a URL."
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": _UA})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        section_map: Dict[int, str] = {}
        current_heading, para_idx = "", 0
        for tag in soup.find_all(True):
            if tag.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                current_heading = tag.get_text(strip=True)
            elif tag.name in ("p", "li", "td", "div", "blockquote", "pre"):
                if not tag.find(["p", "li", "td", "div", "blockquote"]):
                    if current_heading:
                        section_map[para_idx] = current_heading
                    para_idx += 1

        for tag in soup(_NOISE):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True).strip()
        if len(text) < 100:
            return "", {}, ("Page fetched but no usable text extracted. "
                            "The site may require JavaScript. Try Paste Text instead.")
        return text, section_map, ""

    except requests.exceptions.Timeout:
        return "", {}, "Request timed out (15s). Try Paste Text instead."
    except requests.exceptions.ConnectionError:
        return "", {}, "Could not connect. Check the URL and try again."
    except requests.exceptions.HTTPError as exc:
        return "", {}, f"HTTP {exc.response.status_code}: page may require login or does not exist."
    except Exception as exc:
        logger.warning("Unexpected fetch error for %s: %s", url, exc)
        return "", {}, f"Unexpected fetch error: {exc}"
