"""Step 2 — build the external knowledge corpus from pinned Wikipedia revisions.

    python -m src.data.build_corpus [--config config/config.yaml]

What is in the corpus?
  Exactly the English Wikipedia articles that TruthfulQA itself cites in its
  `Source` column (scope `all_truthfulqa`, default: all cited articles, which
  includes distractor articles for questions that were not sampled; or scope
  `sampled_only`). No other source is mixed in.

Reproducibility
  Every article is fetched at the latest revision *at or before*
  `corpus.snapshot_timestamp` via the MediaWiki API; the revision ID is saved
  in data/corpus/manifest.csv. Re-running produces the same text.

Preprocessing (see `html_to_sections`)
  Rendered article HTML -> keep headings, paragraphs, list items and the rows
  of data tables ("wikitable"); drop infoboxes, navigation boxes, reference
  markers ([1]), images, edit links, and everything from the "See also" /
  "References" / "External links" sections onwards.

License: Wikipedia text is CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).

Outputs
  data/corpus/documents.jsonl   one JSON object per article (text + metadata)
  data/corpus/manifest.csv      requested title -> resolved title, revision, status
"""
from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import base_arg_parser, load_config, resolve_path
from src.data.truthfulqa import load_truthfulqa
from src.utils import get_logger, read_jsonl, utc_now, write_jsonl

log = get_logger(__name__)

LICENSE = "CC BY-SA 4.0"
STOP_SECTIONS = {
    "see also", "references", "external links", "further reading", "notes",
    "bibliography", "sources", "citations", "footnotes", "notes and references",
    "works cited", "explanatory notes",
}
REMOVE_SELECTORS = [
    "style", "script", "sup.reference", "sup.noprint", ".mw-editsection", ".navbox",
    ".vertical-navbox", ".infobox", ".sidebar", ".hatnote", ".metadata", ".ambox",
    ".mbox-small", "figure", ".thumb", ".reflist", ".references", ".mw-references-wrap",
    ".noprint", ".toc", ".shortdescription", ".mw-empty-elt", ".gallery", "math",
]
MAX_FAILURE_RATE = 0.10  # abort if more than 10% of articles cannot be fetched


class CorpusError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# HTML -> clean text
# --------------------------------------------------------------------------- #
def _clean(text: str) -> str:
    text = re.sub(r"\[\s*(\d+|[a-z]|citation needed|note \d+)\s*\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _inside(el: Tag, name: str) -> bool:
    return el.find_parent(name) is not None


def html_to_sections(html: str) -> list[dict[str, str]]:
    """Convert rendered Wikipedia HTML into [{'section': ..., 'text': ...}, ...]."""
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one("div.mw-parser-output") or soup
    for selector in REMOVE_SELECTORS:
        for el in root.select(selector):
            el.decompose()

    blocks: list[dict[str, str]] = []
    section = "Introduction"
    for el in root.find_all(["h2", "h3", "h4", "p", "li", "table"]):
        if el.name in ("h2", "h3", "h4"):
            heading = _clean(el.get_text(" "))
            if el.name == "h2" and heading.lower() in STOP_SECTIONS:
                break  # everything after is references / links
            section = heading
            continue
        if el.name == "table":
            if _inside(el, "table") or "wikitable" not in (el.get("class") or []):
                continue
            for tr in el.find_all("tr"):
                cells = [_clean(c.get_text(" ")) for c in tr.find_all(["th", "td"])]
                row = " | ".join(c for c in cells if c)
                if row:
                    blocks.append({"section": section, "text": row})
            continue
        # paragraphs and list items (skip those inside tables / nested lists)
        if _inside(el, "table"):
            continue
        if el.name == "li" and el.find_parent("li") is not None:
            continue
        text = _clean(el.get_text(" "))
        if len(text.split()) >= 3:
            blocks.append({"section": section, "text": text})
    return blocks


# --------------------------------------------------------------------------- #
# Wikipedia API
# --------------------------------------------------------------------------- #
def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = user_agent
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_revision(session: requests.Session, api_url: str, title: str,
                   snapshot: str) -> dict[str, Any] | None:
    """Latest revision of `title` at or before `snapshot` (follows redirects)."""
    params = {
        "action": "query", "format": "json", "formatversion": "2", "redirects": "1",
        "prop": "revisions", "titles": title, "rvlimit": "1", "rvdir": "older",
        "rvstart": snapshot, "rvprop": "ids|timestamp",
    }
    resp = session.get(api_url, params=params, timeout=30)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing") or not pages[0].get("revisions"):
        return None
    page = pages[0]
    rev = page["revisions"][0]
    return {"pageid": page["pageid"], "title": page["title"],
            "revid": rev["revid"], "rev_timestamp": rev["timestamp"]}


def fetch_revision_html(session: requests.Session, api_url: str, revid: int) -> str:
    params = {
        "action": "parse", "format": "json", "formatversion": "2", "oldid": str(revid),
        "prop": "text", "disableeditsection": "1", "disabletoc": "1",
    }
    resp = session.get(api_url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise CorpusError(f"MediaWiki parse error for revid {revid}: {data['error']}")
    return data["parse"]["text"]


# --------------------------------------------------------------------------- #
def corpus_titles(cfg: dict[str, Any]) -> list[str]:
    """The Wikipedia article titles that make up the corpus (sorted, unique)."""
    if cfg["corpus"]["scope"] == "sampled_only":
        path = resolve_path(cfg["dataset"]["processed_path"])
        if not path.is_file():
            raise CorpusError(f"{path} not found — run `python -m src.data.prepare` first.")
        records = read_jsonl(path)
    else:
        records = load_truthfulqa(resolve_path(cfg["dataset"]["raw_path"]))
    return sorted({r["source_wikipedia_title"] for r in records if r["source_wikipedia_title"]})


def main() -> None:
    parser = base_arg_parser(__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if the corpus exists")
    args = parser.parse_args()
    cfg = load_config(args.config)
    c = cfg["corpus"]
    docs_path = resolve_path(c["documents_path"])
    manifest_path = resolve_path(c.get("manifest_path", "data/corpus/manifest.csv"))

    if docs_path.exists() and not args.force:
        log.info("Corpus already exists at %s (use --force to rebuild).", docs_path)
        return

    titles = corpus_titles(cfg)
    log.info("Fetching %d Wikipedia articles (snapshot %s)", len(titles), c["snapshot_timestamp"])
    session = make_session(c["user_agent"])
    docs: dict[int, dict[str, Any]] = {}
    manifest = []

    for i, title in enumerate(titles, start=1):
        entry = {"requested_title": title, "status": "ok", "doc_id": None,
                 "resolved_title": None, "revid": None, "rev_timestamp": None, "error": ""}
        try:
            rev = fetch_revision(session, c["api_url"], title, c["snapshot_timestamp"])
            if rev is None:
                entry.update(status="missing", error="page or revision not found at snapshot")
            else:
                doc_id = f"wiki_{rev['pageid']}"
                entry.update(doc_id=doc_id, resolved_title=rev["title"], revid=rev["revid"],
                             rev_timestamp=rev["rev_timestamp"])
                if rev["pageid"] not in docs:  # several titles can redirect to one page
                    time.sleep(c.get("request_delay_seconds", 0.2))
                    html = fetch_revision_html(session, c["api_url"], rev["revid"])
                    sections = html_to_sections(html)
                    if not sections:
                        entry.update(status="empty", error="no text extracted")
                    else:
                        text = "\n".join(s["text"] for s in sections)
                        docs[rev["pageid"]] = {
                            "doc_id": doc_id, "title": rev["title"],
                            "url": "https://en.wikipedia.org/wiki/" + quote(rev["title"].replace(" ", "_")),
                            "revid": rev["revid"], "rev_timestamp": rev["rev_timestamp"],
                            "snapshot_timestamp": c["snapshot_timestamp"], "license": LICENSE,
                            "retrieved_at": utc_now(), "n_words": len(text.split()),
                            "sections": sections,
                        }
        except (requests.RequestException, CorpusError, KeyError, ValueError) as e:
            entry.update(status="error", error=str(e)[:300])
        if entry["status"] != "ok":
            log.warning("[%d/%d] %s -> %s (%s)", i, len(titles), title, entry["status"], entry["error"])
        elif i % 25 == 0:
            log.info("[%d/%d] fetched", i, len(titles))
        manifest.append(entry)
        time.sleep(c.get("request_delay_seconds", 0.2))

    failed = [m for m in manifest if m["status"] != "ok"]
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    if len(failed) / max(len(titles), 1) > MAX_FAILURE_RATE:
        raise CorpusError(
            f"{len(failed)}/{len(titles)} articles failed (see {manifest_path}). "
            "Aborting instead of building an incomplete corpus; check your connection."
        )
    write_jsonl(docs_path, sorted(docs.values(), key=lambda d: d["doc_id"]))
    log.info("Corpus: %d documents, %d words, %d failed titles -> %s",
             len(docs), sum(d["n_words"] for d in docs.values()), len(failed), docs_path)


if __name__ == "__main__":
    main()
