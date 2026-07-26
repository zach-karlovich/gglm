"""Search DTIC public technical reports and download PDFs. No API key needed.

DTIC itself 403s every scripted client, so this goes through the Internet
Archive mirror of the public-release collection (collection:dticarchive).
"""

import os
import re
import sys
import time
from pathlib import Path

import httpx

from gglm import catalog
from gglm.parse.pdf import file_sha256

SEARCH = "https://archive.org/advancedsearch.php"
DOWNLOAD = "https://archive.org/download"
DETAILS = "https://archive.org/details"
DTIC_CITATIONS = "https://apps.dtic.mil/sti/citations"
CONTACT_EMAIL = "ddf4me@virginia.edu"
HEADERS = {"User-Agent": f"gglm/0.1 (mailto:{CONTACT_EMAIL})"}
PAGE_SIZE = 100
SLEEP_SECONDS = 2
DATA = Path(os.environ.get("GGLM_DATA", "data"))


RETRIES = 3 # archive.org CDN nodes 500 routinely; one killed a whole collect run


def ad_number(identifier):
    """'DTIC_ADA510750' -> 'ADA510750'."""
    return identifier.removeprefix("DTIC_")


def get(url, **kwargs):
    """GET with retries; returns None when a transient failure won't quit."""
    for attempt in range(RETRIES):
        try:
            r = httpx.get(url, headers=HEADERS, follow_redirects=True, **kwargs)
        except httpx.RequestError as e:
            print(f"  {type(e).__name__} on {url}")
        else:
            if r.status_code < 500:
                return r
            print(f"  {r.status_code} on {url}")
        if attempt < RETRIES - 1:
            time.sleep(SLEEP_SECONDS * (attempt + 1))
    return None


def search(query, max_records=200):
    """Yield mirror records for DTIC reports matching the query phrase."""
    page = 1
    yielded = 0
    while True:
        params = {
            "q": f'collection:dticarchive AND "{query}"',
            "fl[]": ["identifier", "title", "year", "creator"],
            "rows": PAGE_SIZE,
            "page": page,
            "output": "json",
        }
        r = get(SEARCH, params=params, timeout=30)
        if r is None:
            return
        r.raise_for_status()
        docs = r.json()["response"]["docs"]
        if not docs:
            return

        for record in docs:
            yield record
            yielded += 1
            if yielded >= max_records:
                return

        if len(docs) < PAGE_SIZE:
            return
        page += 1
        time.sleep(SLEEP_SECONDS)


def download_pdf(record, out_dir=DATA / "raw" / "dtic"):
    """Save a record's PDF as {ad}.pdf and return the path, or None."""
    identifier = record["identifier"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ad_number(identifier)}.pdf"
    if out_path.exists():
        return out_path
    url = f"{DOWNLOAD}/{identifier}/{identifier}.pdf"
    r = get(url, timeout=120)
    if r is None or r.status_code != 200:
        return None  # missing PDF derivative, or the CDN kept failing
    if "pdf" not in r.headers.get("content-type", ""):
        return None
    out_path.write_bytes(r.content)
    time.sleep(SLEEP_SECONDS)
    return out_path


def clean_title(record):
    """Mirror titles repeat the AD number: 'DTIC ADA510750: <title>'."""
    title = " ".join(str(record.get("title", "")).split())
    return re.sub(r"^DTIC\s+AD\S*:\s*", "", title)


def catalog_entry(record, query):
    """Map a mirror record to a catalog entry. sha256/kind filled in later."""
    identifier = record["identifier"]
    ad = ad_number(identifier)
    creators = record.get("creator", [])
    if isinstance(creators, str):
        creators = [creators]
    return {
        "key": f"dtic:{ad}",
        "source": "dtic",
        "title": clean_title(record),
        "authors": creators,
        "year": str(record.get("year", "")),
        "sti_type": "report",
        "report_numbers": [ad],
        "license": None,  # collection mirrors DTIC approved-for-public-release
        "citation_url": f"{DETAILS}/{identifier}",
        "dtic_url": f"{DTIC_CITATIONS}/{ad}",
        "query": query,
        "retrieved": time.strftime("%Y-%m-%d"),
        "pdf": None,
        "sha256": None,
        "kind": None,
    }


if __name__ == "__main__":
    # usage: python -m gglm.sources.dtic "<query>" [max] [--download]
    if len(sys.argv) < 2:
        print('usage: python -m gglm.sources.dtic "<query>" [max] [--download]')
        sys.exit(1)

    query = sys.argv[1]
    max_records = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 25
    do_download = "--download" in sys.argv

    known = catalog.load()
    n_new = 0
    for record in search(query, max_records):
        key = f"dtic:{ad_number(record['identifier'])}"
        entry = catalog_entry(record, query)

        if do_download:
            path = download_pdf(record)
            if path:
                entry["pdf"] = str(path)
                entry["sha256"] = file_sha256(path)

        if key not in known:
            catalog.append(entry)
            known[key] = entry
            n_new += 1
        elif entry["pdf"] and not known[key].get("pdf"):
            known[key] = catalog.update(key, pdf=entry["pdf"], sha256=entry["sha256"])

        flag = "pdf" if known[key].get("pdf") else "---"
        print(f"{key:<22} {entry['year']:4}  {flag:3}  {entry['title'][:60]}")

    print(f"{n_new} new entries -> {catalog.CATALOG}")
