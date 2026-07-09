"""Search OSTI.gov and download public full-text PDFs. No API key needed.

API: https://www.osti.gov/api/v1/records, paginated 1-based. Full text comes
from the PURL link, present only on publicly released documents. Every record
found is registered in the catalog (gglm.catalog).
"""

import os
import sys
import time
from pathlib import Path

import httpx

from gglm import catalog
from gglm.parse.pdf import file_sha256

API = "https://www.osti.gov/api/v1/records"
BIBLIO = "https://www.osti.gov/biblio"
CONTACT_EMAIL = "ddf4me@virginia.edu"
HEADERS = {"User-Agent": f"gglm/0.1 (mailto:{CONTACT_EMAIL})"}
PAGE_SIZE = 100
SLEEP_SECONDS = 2
DATA = Path(os.environ.get("GGLM_DATA", "data"))


def fulltext_url(record):
    """The PURL full-text link, or None."""
    for link in record.get("links", []):
        if link.get("rel") == "fulltext":
            return link["href"]
    return None


def search(query, max_records=200):
    """Yield records that have a full-text link."""
    page = 1
    yielded = 0
    while True:
        params = {"q": query, "rows": PAGE_SIZE, "page": page}
        r = httpx.get(API, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        results = r.json()
        if not results:
            return

        for record in results:
            if fulltext_url(record) is None:
                continue
            yield record
            yielded += 1
            if yielded >= max_records:
                return

        if len(results) < PAGE_SIZE:
            return
        page += 1
        time.sleep(SLEEP_SECONDS)


def download_pdf(record, out_dir=DATA / "raw" / "osti"):
    """Save a record's full text as {osti_id}.pdf and return the path.
    Returns None if the PURL serves something other than a PDF."""
    url = fulltext_url(record)
    if url is None:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{record['osti_id']}.pdf"
    if out_path.exists():
        return out_path
    r = httpx.get(url, headers=HEADERS, timeout=120, follow_redirects=True)
    r.raise_for_status()
    if "pdf" not in r.headers.get("content-type", ""):
        return None  # some PURLs serve HTML landing pages
    out_path.write_bytes(r.content)
    time.sleep(SLEEP_SECONDS)
    return out_path


def author_name(author):
    """'Ao, Tommy [Sandia ...] (ORCID:...)' -> 'Ao, Tommy'."""
    return author.split(" [")[0].strip()


def catalog_entry(record, query):
    """Map an OSTI record to a catalog entry. kind filled in after parsing."""
    return {
        "key": f"osti:{record['osti_id']}",
        "source": "osti",
        "title": " ".join(record["title"].split()),
        "authors": [author_name(a) for a in record.get("authors", [])],
        "year": record.get("publication_date", "")[:4],
        "sti_type": record.get("product_type"),
        "report_numbers": [n for n in [record.get("report_number")] if n],
        "doi": record.get("doi"),
        "license": None,  # OSTI has no license field; the PURL implies public release
        "citation_url": f"{BIBLIO}/{record['osti_id']}",
        "query": query,
        "retrieved": time.strftime("%Y-%m-%d"),
        "pdf": None,
        "sha256": None,
        "kind": None,
    }


if __name__ == "__main__":
    # usage: python -m gglm.sources.osti "<query>" [max] [--download]
    if len(sys.argv) < 2:
        print('usage: python -m gglm.sources.osti "<query>" [max] [--download]')
        sys.exit(1)

    query = sys.argv[1]
    max_records = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 25
    do_download = "--download" in sys.argv

    known = catalog.load()
    n_new = 0
    for record in search(query, max_records):
        key = f"osti:{record['osti_id']}"
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
        print(f"{key:<22} {entry['year']:4}  {flag:3}  {str(entry['sti_type']):<20} {entry['title'][:60]}")

    print(f"{n_new} new entries -> {catalog.CATALOG}")
