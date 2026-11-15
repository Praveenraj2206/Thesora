"""N1: dump extracted text per page and flag likely two-column pages (interleaving risk).

Usage: python scripts/check_extraction.py [--docs data/papers] [--dump]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.ingestion.layout import column_report, table_chunks  # noqa: E402
from app.ingestion.pdf_loader import load_pdf  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default=str(config.PAPERS_DIR))
    parser.add_argument("--dump", action="store_true", help="print each page's extracted text")
    args = parser.parse_args()
    for pdf in sorted(Path(args.docs).glob("*.pdf")):
        pages = load_pdf(pdf, pdf.name)
        report = column_report(pdf)
        two = [r["page"] for r in report if r["two_column"]]
        tables = table_chunks(pdf, pdf.name, 0, {})
        print(f"{pdf.name}: {len(pages)} pages, two-column pages: {two or 'none'}, tables: {len(tables)}")
        if args.dump:
            for page in pages:
                print(f"--- p.{page.page_number}\n{page.text[:1500]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
