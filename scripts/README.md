# Screening scripts (how corpus v1 was built)

Run in this order on a folder of candidate PDFs (pypdf 6.21.2, wordfreq 3.1.1; no OCR):

1. `pagescore2.py out.jsonl *.pdf` — one row per PDF page: Arabic tokens, damaged-word rate (tokens unknown to the
   wordfreq Arabic lexicon), Arabic-letter share, replacement characters, swapped lam-alef, reversed word order; `keep` flag.
2. `summarize2.py out.jsonl` — document decision: accept if kept pages average ≤ 3 % damaged words and word order is not reversed.
3. `hiddencheck.py file.pdf ...` — compares pypdf with PyMuPDF's visible-page text; a ratio far from 1 means hidden,
   off-page or missing text (two documents were rejected this way).
4. `build_passages.py docs_summary.json pdf_dir passages.jsonl scores.jsonl ...` — page-bounded ~120-word passages.

`src/bayan_preprocess.py` re-implements step 4 and the cleaning pipeline used by the notebook.
