"""N1: dump extracted text per page and flag likely two-column pages (interleaving risk).

Usage: python scripts/check_extraction.py [--docs data/papers] [--dump] [--dropped]

--dropped lists what the clean-extraction rules removed (headers, authors, figure labels, ...) per paper,
so you can check that nothing useful was thrown away.
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


def show_dropped(pdf: Path) -> None:
    import pymupdf
    from collections import Counter

    from app.ingestion.clean_layout import clean_blocks, document_layout
    from app.ingestion.layout import detect_tables

    with pymupdf.open(str(pdf)) as doc:
        raws = [page.get_text("rawdict") for page in doc]
        layout = document_layout(doc, raws)
        layout.tables = detect_tables(doc, pdf)
        dropped: list[tuple[str, str]] = []
        for i, (page, raw) in enumerate(zip(doc, raws)):
            clean_blocks(page, raw, i, layout, dropped)
    counts = Counter(reason for reason, _ in dropped)
    print("   dropped:", ", ".join(f"{reason} {n}" for reason, n in counts.most_common()) or "nothing")
    for reason in counts:
        examples = [" ".join(t.split())[:70] for r, t in dropped if r == reason][:2]
        print(f"     {reason}: " + " | ".join(examples))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default=str(config.PAPERS_DIR))
    parser.add_argument("--dump", action="store_true", help="print each page's extracted text")
    parser.add_argument("--dropped", action="store_true", help="show what clean extraction removed, by reason")
    args = parser.parse_args()
    for pdf in sorted(Path(args.docs).glob("*.pdf")):
        pages = load_pdf(pdf, pdf.name)
        report = column_report(pdf)
        two = [r["page"] for r in report if r["two_column"]]
        tables = table_chunks(pdf, pdf.name, 0, {})
        print(f"{pdf.name}: {len(pages)} pages, two-column pages: {two or 'none'}, tables: {len(tables)}")
        if args.dropped:
            show_dropped(pdf)
        if args.dump:
            for page in pages:
                print(f"--- p.{page.page_number}\n{page.text[:1500]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
