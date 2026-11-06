# Bayan (بيان) — Arabic PDF Question Answering

Bayan answers Arabic questions from a fixed collection of Arabic PDF documents, cites the document and page it used, and declines when the documents do not contain the answer. Text comes **only from the PDFs' embedded text layer — no OCR, no vision models**.

University NLP course project. Current stage: **Milestone 1 — dataset, EDA and preprocessing pipeline**.

[Open the Milestone 1 notebook in Colab](https://colab.research.google.com/github/AhmedElkhateeb11/bayan-arabic-pdf-qa/blob/main/notebooks/Bayan_Milestone1.ipynb)

## Corpus v1 at a glance

| | |
|---|---|
| Documents | 24 born-digital Arabic PDFs (40 screened, 16 rejected) |
| Publishers | WHO / WHO EMRO (14), UNESCO (4), WIPO (5), ITU (1) |
| Pages | 2,790 PDF pages → 2,228 kept after screening |
| Passages | **6,729** page-bounded passages of ~120 words (the dataset instance) |
| Words | ~803,000 |
| Language | Modern Standard Arabic |
| Labels | none in the sources; an independent evaluation set (~300 questions) is in preparation |
| Licences | CC BY / CC BY-SA / CC BY-NC-SA 3.0 IGO, per document — see [DATA_LICENSES.md](DATA_LICENSES.md) |

Screening rule (pypdf 6.21.2): a page is kept if it has ≥ 40 Arabic words, ≥ 60 % Arabic letters, ≤ 5 % words unknown to the `wordfreq` Arabic lexicon, ≤ 3 replacement characters, no swapped lam-alef and no reversed word order. A document is accepted if its kept pages average ≤ 3 % unknown words, its word order is not reversed, and pypdf reads the same visible text as PyMuPDF.

## Repository layout

```
data/corpus_v1/
  pdf/                    the 24 accepted PDFs (licences in DATA_LICENSES.md)
  manifest_v1.csv         all 40 screened PDFs: source URL, licence, SHA-256, counts, decision and reason
  passages_v1.jsonl       6,729 passages: passage_id, file, pdf_page, words, exact_dup_of, text
  page_scores_v1.jsonl    page-level screening measures for all 6,033 screened pages
src/bayan_preprocess.py   extraction, passage rebuild and the 10-step preprocessing pipeline
notebooks/                Bayan_Milestone1.ipynb (+ .py source, jupytext)
scripts/                  the screening scripts used to build corpus v1
```

`passage_id` = `<document_id>_p<pdf page, 4 digits>_<chunk index>`, e.g. `who_10665_383109_p0060_0`. Passage IDs and PDF page numbers are frozen: the evaluation set refers to them.

## Run locally

```bash
pip install -r requirements.txt
jupyter notebook notebooks/Bayan_Milestone1.ipynb
```

The notebook verifies every PDF by SHA-256, re-extracts the text, rebuilds all 6,729 passages (identical to `passages_v1.jsonl`), runs the EDA and writes `data/processed/passages_v1_processed.parquet` with three views per passage: `clean_text` (for reading and answering), `retrieval_text` (orthographically normalised) and `tfidf_text` (stop words removed, light-stemmed).

