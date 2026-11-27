"""Retrieval-level check of TEST_PLAN_SURVEY_PAPERS.md on the six survey papers (no LLM needed).

Usage: python scripts/survey_check.py "C:/thesora_survey"

For each test question it checks that the passage holding the answer is among the passages the model
would see (or that Thesora abstains for the out-of-scope questions), then runs Compare coverage, three
Agree? topics and prints the glossary size. Takes about 2 minutes on a laptop CPU.
"""
import sys, time, re, warnings, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from app import config
from app.pipelines.rag import ResearchAssistant
D = sys.argv[1]
a = ResearchAssistant(); t = time.time(); a.ingest_folder(D); print(f"ingest {time.time()-t:.0f}s, {len(a.store)} chunks")
Q = [  # question, regex the answer passage must contain, expect_abstain
 ("What is omni-moderation-latest used for?", r"omni-moderation", False),
 ("Which moderation model screens user queries in Ai2 Scholar QA?", r"omni-moderation", False),
 ("What value of K is used in Reciprocal Rank Fusion?", r"K\s*=\s*60", False),
 ("Which embedding model does PaperQA use?", r"text-embedding-ada-002", False),
 ("What relevance threshold does DeepEra use to filter passages?", r"τ\s*=\s*0\.8|0\.8\)", False),
 ("How often does GPT-4o hallucinate citations according to OpenScholar?", r"78.90", False),
 ("What citation accuracy and single-pass success rate does HySemRAG report?", r"99\.0", False),
 ("What is the price of an NVIDIA H100 GPU?", None, True),
 ("How do these systems handle speech or audio input?", None, True),
 ("What nDCG@10 does mxbai-rerank-large-v1 achieve?", r"0\.927", False),
 ("What accuracy and precision did PaperQA get on LitQA?", r"69\.5", False),
 ("What disease and exposure did HySemRAG apply its framework to?", r"ozone", False),
 ("What chunk size does PaperQA use?", r"4,000", False),
 ("How is LFS computed?", r"Logic\s+Fidelity\s+Score|0\s*to\s*5", False),
]
ok = 0
for q, rx, abstain in Q:
    chunks = a.retrieve_for(q)
    best = max((c.relevance for c in chunks), default=0.0)
    would_abstain = best < config.ABSTAIN_THRESHOLD
    ctx = [a._context_text(c.chunk, n) for n, c in enumerate(chunks, 1)]
    hit = next((n for n, t in enumerate(ctx, 1) if rx and re.search(rx, t)), None)
    good = would_abstain if abstain else (hit is not None and not would_abstain)
    ok += good
    print(f"{'PASS' if good else 'FAIL'} best={best:.2f} hit@{hit} {'(abstains)' if would_abstain else ''} | {q}")
    if not good and not abstain:
        for n, c in enumerate(chunks, 1): print(f"     [{n}] {c.relevance:.2f} {c.chunk.doc_id[:18]} p{c.chunk.page_number} {c.chunk.text[:90]!r}")
print(f"\n{ok}/{len(Q)} retrieval checks pass")
print("paper names:", {d[:14]: sorted(n, key=n.get, reverse=True)[:4] for d, n in a.paper_names.items()})
t = time.time(); rows = a.retriever.retrieve_across_documents("Compare how these papers retrieve and rerank evidence")
print(f"\nCompare covers {len({c.chunk.doc_id for c in rows})}/6 papers ({time.time()-t:.1f}s)")
for topic in ["citation hallucination", "citation accuracy", "reranker model"]:
    r = a.agree(topic)
    print(f"\nAgree? '{topic}': {len(r['rows'])} papers; conflicts sev>1: {[c['reason'] for c in r['conflicts'] if c['severity']>1][:3]}")
    for row in r["rows"]: print(f"   {row['label']}: {row['position'][:110]!r}")
print("\nGlossary:", len(a.glossary), "terms;", [(g['abbr'], len(g.get('papers', []))) for g in a.glossary][:25])
