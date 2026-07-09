"""Search NASA NTRS and download public full-text PDFs. No API key needed.

Rate limit: 500 requests per 15 min. Docs: https://ntrs.nasa.gov/api/openapi/
Every record found is registered in the catalog (gglm.catalog).
"""

import sys
import time
from pathlib import Path

import httpx

from gglm import catalog
from gglm.parse.pdf import file_sha256

HOST = "https://ntrs.nasa.gov"
API = f"{HOST}/api"
CONTACT_EMAIL = "ddf4me@virginia.edu"
HEADERS = {"User-Agent": f"gglm/0.1 (mailto:{CONTACT_EMAIL})"}
PAGE_SIZE = 100
SLEEP_SECONDS = 2

SKIP_STI_TYPES = {"ABSTRACT", "EXTENDED_ABSTRACT", "VIDEO"}


def search(query, max_records=200):
    """Yield public, downloadable records, skipping SKIP_STI_TYPES."""
    offset = 0
    yielded = 0
    while True:
        params = {
            "q": query,
            "distribution": "PUBLIC",
            "disseminated": "DOCUMENT_AND_METADATA",
            "page.size": PAGE_SIZE,
            "page.from": offset,
        }
        r = httpx.get(f"{API}/citations/search", params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data["results"]
        if not results:
            return

        for record in results:
            if record.get("stiType") in SKIP_STI_TYPES:
                continue
            yield record
            yielded += 1
            if yielded >= max_records:
                return

        offset += len(results)
        if offset >= data["stats"]["total"]:
            return
        time.sleep(SLEEP_SECONDS)


def pdf_url(record):
    """PDF link from a search result, or None. links.pdf also covers
    documents NTRS converted to PDF (e.g. pptx originals)."""
    for item in record.get("downloads", []):
        link = item.get("links", {}).get("pdf")
        if link:
            return HOST + link
    return None


def download_pdf(record, out_dir="data/raw/ntrs"):
    """Save a record's PDF as {id}.pdf and return the path, or None if no PDF."""
    url = pdf_url(record)
    if url is None:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{record['id']}.pdf"
    if out_path.exists():
        return out_path
    r = httpx.get(url, headers=HEADERS, timeout=120, follow_redirects=True)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    time.sleep(SLEEP_SECONDS)
    return out_path


def record_year(record):
    """Publication year, falling back to distribution date."""
    pub_date = (record.get("publications") or [{}])[0].get("publicationDate")
    return (pub_date or record.get("distributionDate", ""))[:4]


def catalog_entry(record, query):
    """Map an NTRS record to a catalog entry. sha256/kind filled in later."""
    authors = [
        a.get("meta", {}).get("author", {}).get("name", "")
        for a in record.get("authorAffiliations", [])
    ]
    report_numbers = [
        n for n in record.get("otherReportNumbers", [])
        if not n.startswith("Report Number:")  # NTRS lists these twice
    ]
    return {
        "key": f"ntrs:{record['id']}",
        "source": "ntrs",
        "title": " ".join(record["title"].split()),
        "authors": [a for a in authors if a],
        "year": record_year(record),
        "sti_type": record.get("stiType"),
        "report_numbers": report_numbers,
        "license": record.get("copyright", {}).get("determinationType"),
        "citation_url": f"{HOST}/citations/{record['id']}",
        "query": query,
        "retrieved": time.strftime("%Y-%m-%d"),
        "pdf": None,
        "sha256": None,
        "kind": None,  # set after parsing
    }


if __name__ == "__main__":
    # usage: python -m gglm.sources.ntrs "<query>" [max] [--download]
    if len(sys.argv) < 2:
        print('usage: python -m gglm.sources.ntrs "<query>" [max] [--download]')
        sys.exit(1)

    query = sys.argv[1]
    max_records = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 25
    do_download = "--download" in sys.argv

    known = catalog.load()
    n_new = 0
    for record in search(query, max_records):
        key = f"ntrs:{record['id']}"
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
        print(f"{key:<22} {entry['year']:4}  {flag:3}  {entry['sti_type']:<20} {entry['title'][:60]}")

    print(f"{n_new} new entries -> {catalog.CATALOG}")
