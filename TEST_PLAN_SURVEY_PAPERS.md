# Thesora v2: test plan for all 18 features (survey papers)

Every expected answer below was checked against the papers. If Thesora says something the "Pass looks like" column doesn't, check the cited page before you call it a pass.

---

## Step 0: Prepare the 6 PDFs (5 minutes)

1. Make a folder **outside the repo** so the PDFs never get committed, e.g. `C:\thesora_survey\`.
2. Put the 6 PDFs there with these names. Two of the markdown files you sent had swapped names, so double-check which paper is which:

| File name | Paper | Check on page 1 |
|---|---|---|
| `OpenScholar.pdf` | Asai et al., *Nature* 2026 | "Synthesizing scientific literature with retrieval-augmented language models" |
| `Agentic_Hybrid_RAG.pdf` | Nagori et al., 2025 | "Open-Source Agentic Hybrid RAG Framework for Scientific Literature Review" |
| `Ai2_Scholar_QA.pdf` | Singh et al., ACL 2025 Demo | "Ai2 Scholar QA: Organized Literature Synthesis with Attribution" |
| `DeepEra.pdf` | Chen et al., 2026 | "DeepEra: A Deep Evidence Reranking Agent…" |
| `HySemRAG.pdf` | Godinez, 2025 | "HySemRAG: A Hybrid Semantic RAG Framework…" |
| `PaperQA.pdf` | Lála et al., 2023 | "PaperQA: Retrieval-Augmented Generative Agent for Scientific Research" |

3. Check the text extracts properly (no Ollama needed). In Git Bash, from the Thesora folder with `.venv` active:

```bash
python scripts/check_extraction.py --docs "C:/thesora_survey"
```

**Pass:** every PDF shows pages > 0, most of them two-column, and some tables found (Ai2 and PaperQA have the clearest ones). If a PDF says "no text" or "scanned", tell me which one.

4. After pulling new code, always open a **new chat** and upload the PDFs again: the text is extracted at upload.
5. Start the app as usual (terminal 1: `uvicorn`, terminal 2: `streamlit`). Open a **new chat** and upload all 6 PDFs in one go.

---

## Step 1: The 5 Phase-1 features

### F1 · Hybrid retrieval (keywords + meaning)
**Why:** BM25 finds exact rare strings, and embeddings find reworded questions. One question of each kind shows both halves work.

| # | Ask (Ask a question mode) | Pass looks like |
|---|---|---|
| Q1 | `What is omni-moderation-latest used for?` | Ai2 Scholar QA uses it for **query validation**, to block harmful queries before processing. Cites Ai2. *(Exact rare term: the keyword half.)* |
| Q2 | `Which moderation model screens user queries in Ai2 Scholar QA?` | **omni-moderation-latest** (OpenAI), found from a reworded question. *(Meaning half.)* Known hard case: `How does Ai2 Scholar QA stop harmful questions before answering them?` refuses — the PDF splits that sentence across two columns, so no single passage holds it. Refusing is the honest result, not a wrong answer. |
| Q3 | `What value of K is used in Reciprocal Rank Fusion?` | **K = 60** (HySemRAG). Worth showing in the demo: Thesora uses the same RRF constant. |

### F2 · Cross-encoder reranking
**Why:** several papers describe similar things (embedding models, thresholds). The reranker has to put the right paper's passage first.

| # | Ask | Pass looks like |
|---|---|---|
| Q4 | `Which embedding model does PaperQA use?` | **text-embedding-ada-002** (PaperQA). Source [1] must be PaperQA, **not** Agentic RAG (all-MiniLM-L6-v2) or Ai2 (mxbai-embed-large-v1). |
| Q5 | `What relevance threshold does DeepEra use to filter passages?` | **τ = 0.8**, from DeepEra. Open the sources: the top one should have the highest `rel`. |

### F3 · Citation verification (✓ / !)
**Why:** numbers are where models slip. The number guard has to catch any number that isn't in the cited passage.

| # | Ask | Pass looks like |
|---|---|---|
| Q6 | `How often does GPT-4o hallucinate citations according to OpenScholar?` | **78–90%** of the time. Sources ✓ Verified. |
| Q7 | `What citation accuracy and single-pass success rate does HySemRAG report?` | **99.0%** citation accuracy, **68.3%** single-pass success. Every number must be ✓. If one shows **!**, open the passage and check whether the number is really there. |

### F4 · Abstention + confidence
**Why:** the most important trust feature. Thesora should refuse when the papers don't say something, not guess.

| # | Ask | Pass looks like |
|---|---|---|
| Q8 | `What is the price of an NVIDIA H100 GPU?` | **Refuses.** DeepEra mentions H100 GPUs but never a price; this is a good trap. |
| Q9 | `How do these systems handle speech or audio input?` | **Refuses** (none of the papers cover it). |
| Q10 | Any answered question above | The confidence meter shows a value; Q6/Q7 should be high. |

### F5 · Evaluation harness (terminal)
**Why:** shows the numbers come from a script, not from opinion.

```bash
python scripts/evaluate.py --limit 3 --no-judge
```

```bash
python scripts/compare_retrievers.py
```

**Pass:** both finish and write their JSON files. Note: these use the bundled DejaVu sample papers and their golden questions. A golden set for the 6 survey papers can be added later (see the end).

### Extras
| Feature | Do this | Pass looks like |
|---|---|---|
| Cross-paper synthesis | **Compare papers** mode: `Compare how these papers retrieve and rerank evidence` | One row per paper (6 rows) + a **Research Gaps** section. All 6 papers appear. |
| Isolated chats | Open a **new chat** (no uploads) and ask Q4 | "No documents in this chat yet". The other chat's papers are not visible. |
| Upload limit | Try uploading 11 PDFs at once | Refused with the 10-per-upload message. |
| Prompt refiner | Turn on **✨ Prompt refiner**, type `deepera reranking how good` → **✨ Refine** | A precise question appears (e.g. about DeepEra's improvement over other rerankers). **Use suggestion** → answer mentions **up to 8% relative improvement**. |
| Interface | Toggle the dark theme; scroll down | The mode bar stays pinned; the theme switches cleanly. |

---

## Step 2: The 13 v2 features

### N1 · Tables become citable passages
**Why:** benchmark results in papers sit in tables; earlier they were lost.

| # | Ask | Pass looks like |
|---|---|---|
| Q11 | `What nDCG@10 does mxbai-rerank-large-v1 achieve?` | **0.927** (Ai2, Table 1). Open the source: it shows table rows (`|` columns). |
| Q12 | `What accuracy and precision did PaperQA get on LitQA?` | **69.5%** accuracy, **87.9%** precision (PaperQA, Table 2). |

If a table answer abstains, check that table was listed in Step 0 output. If it wasn't, table detection missed it; tell me which one.

### N2 · Search upgrades (context expansion on; the other two off)
**Why:** context expansion is now on for the 2 best passages; Q13 (ozone) and Q16 (LFS) depend on it.
- Restart the backend with `MULTI_QUERY=1` and ask Q2 again. **Pass:** the same answer, slightly slower. Then turn it back off.
- (Optional) `CONTEXT_EXPANSION=0`, then ask Q16 again: it may now refuse, which shows why expansion is on.

### N3 · Comparison-table cells verified
After the Compare run (Extras), open **⋯ → Table check**.
**Pass:** each cell shows ✓ / ~ / ✗ / –, with a page number. Spot-check one: Agentic Hybrid RAG's reranker cell should say **Cohere rerank-english-v3.0**; Ai2's should say **mxbai-rerank-large-v1**.

### N4 · Real citations + BibTeX + notes
**Why:** sources should look like a paper's references, not file names.
**Pass:** source lines read like `Asai et al., 2026 · p.N · Section`. In the sidebar, **Download .bib** has 6 entries and **Download notes (.md)** lists only verified citations. Wrong author/year on any paper → note which one (auto-detection; can be corrected with the PATCH endpoint).

### N5 · Per-sentence check + silent papers
| # | Ask | Pass looks like |
|---|---|---|
| Q13 | `What disease and exposure did HySemRAG apply its framework to?` | **Ozone exposure and cardiovascular disease** (geospatial epidemiology). Under the answer: "No evidence from: …the other papers… — silence is not disagreement". |
| | ⋯ → **Sentence check** on Q7 | Each sentence shows ✓ / ~ / ! / ○ with text. |

### N6 · Partial answers
**Why:** one half of a question being unanswerable shouldn't block the half that is answerable.

| # | Ask | Pass looks like |
|---|---|---|
| Q14 | `What chunk size does PaperQA use? What is the price of an NVIDIA H100 GPU?` | Part 1: **overlapping 4,000-character chunks**. Part 2: "Not found in your papers". Two part chips. |

### N7 · Large files
The survey PDFs are probably under 5 MB. Upload any PDF **over 5 MB** in a fresh chat. **Pass:** a size warning appears, but it still indexes.

### N8 · Meaning check (off)
**Pass with it off:** the UI says checking is by wording only.
Optional: restart the backend with `NLI_ENABLED=1` (downloads 283 MB once) and ask Q6. Sources then show a meaning label. Turn it off afterwards.

### N9 · PDF side panel with highlight
On Q11's answer, open source [1] → **📄 Open page**. **Pass:** the cited page opens on the right with the passage outlined; it stays in view as you scroll. **✕ Close panel** hides it.

### N10 · Self-repair
**Why:** if a sentence fails the check, Thesora retries once and removes what is still unsupported.

| # | Ask | Pass looks like |
|---|---|---|
| Q15 | `List every system in the LitQA table with its exact accuracy and precision` | Either everything is ✓, **or** a note "N claims removed". Both are passes; a wrong number shown as ✓ is a fail. |

It can't be forced on demand. The unit tests prove the logic: `python -m pytest tests/test_trust.py -q`.

### N11 · Explain + glossary
- On Q3's source → **💡 Explain**. **Pass:** What / Why / How / Example, labelled "Plain-English explanation — not evidence".
- In the sidebar, **Glossary**. **Pass:** abbreviations with long forms, e.g. **NRS** (Noise Robustness Score), **CDR** (Context Discrimination Rate), **LFS** (Logic Fidelity Score), **DPO** (Direct Preference Optimization), **RAG**.
- Q16 `How is LFS computed?` **Pass:** DeepEra's **Logic Fidelity Score**, an LLM-based **0 to 5** score for factual correctness, logical consistency and contextual adequacy. *(The short form matches because the glossary adds the long form to keyword search.)*

### N12 · Agree? mode
**Why:** these papers report different numbers on the same topic, so it's a real test.

**Run each topic on its own** (type one topic, press Enter, then the next).

| # | Topic (Agree? mode) | Pass looks like |
|---|---|---|
| Q17 | `citation hallucination` | Quotes from PaperQA (how hallucinated citations were categorised), OpenScholar (GPT-5 still fabricates citations) and Ai2. Each quote is word for word with a page. |
| Q18 | `citation accuracy` | 4 papers: HySemRAG (field-level accuracy, 74.4% DOI matching …), OpenScholar, PaperQA, Ai2. |
| Q19 | `reranker model` | OpenScholar, Ai2 (its reranker table) and DeepEra. A "⚠️ Numbers differ" here compares model sizes, so read both quotes. |
| Q20 | `cost per question` | **Only one paper** (PaperQA, $0.18): Thesora should say "Only one paper discusses this topic", not "no conflicts". |

No quote should come from a References section. Note: these compare different setups, so "numbers differ" means "look closer", not "one paper is wrong". Say that in the demo.

### N13 · Stress test (terminal)
```bash
python scripts/stress_eval.py --limit 3
```
**Pass:** it prints citation recall, citation precision, distractor-cited rate and altered-numbers-repeated rate. It uses the built-in cases. DeepEra's SSLI distractors ("semantically similar but logically irrelevant") are the same idea, which is a good link for the report.

---

## Quick check without the model (terminal)

```bash
python scripts/survey_check.py "C:/thesora_survey"
```
Expect 14/14 retrieval checks to pass, Compare 6/6 papers, and paper names such as "Lála et al., 2023" and "Godinez, 2025".

## Step 3: Record the results

For each Q, note **Pass / Fail / Odd**. For any Fail or Odd, send me:
- the question,
- a screenshot of the answer with its sources opened,
- any red error from either terminal.

All tests together: `python -m pytest -q` (expect 68 passed).

## Possible "perfect v2" improvements after testing
- A golden QA set for these 6 papers (`data/eval/survey_qa.json`), so `evaluate.py --docs C:/thesora_survey --golden data/eval/survey_qa.json` reports real numbers on your survey papers.
- Fixes for whatever fails above.
