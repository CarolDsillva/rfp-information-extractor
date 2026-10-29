# RFP Information Extractor

Turn messy Request for Proposal (RFP) documents — PDFs, HTML bid pages, addenda — into clean, structured JSON using Claude and retrieval-augmented generation (RAG).

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![LLM](https://img.shields.io/badge/LLM-Claude-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> Built as my solution to the **Emplay Analytics Campus Hiring 2026** technical assignment.

---

## The problem

Public-sector bids are spread across a main RFP (often 60+ pages), several addenda, affidavits, spec sheets and a bid-portal web page. A sales team has to read all of it to answer basic questions: *When is it due? Is there a bid bond? Which models are required? Who do we contact?*

The assignment: build a program that parses HTML and PDF bid documents and uses language models to map their contents onto a fixed set of fields, outputting the result as JSON.

## How it works

```
 PDF / HTML ──► Parse ──► (large doc?) ──► Retrieve ──► Extract ──► Validate ──► JSON
               pypdf       > 50K chars     MiniLM        Claude      21-field
               BeautifulSoup               embeddings                schema
```

1. **Parse** – `pypdf` for PDFs; `BeautifulSoup` for HTML (scripts, styles and markup noise stripped).
2. **Retrieve (RAG)** – documents over 50K characters are split into overlapping 400-word chunks and embedded with `all-MiniLM-L6-v2`. Instead of one generic query, the retriever runs **18 field-specific queries** (bid bond, payment terms, pre-bid meeting, evaluation criteria, …), so details buried deep in a long RFP still reach the model. The most relevant chunks are packed into a 60K-character budget and re-ordered to follow the original document.
3. **Extract** – Claude (temperature 0) maps the text onto the schema using a system prompt plus a field-by-field guide.
4. **Validate** – the response is parsed defensively (handles markdown fences and stray text), unknown keys are dropped, lists are flattened, and every record is guaranteed to have all 21 fields.

## Extracted fields

| Group | Fields |
|---|---|
| Core | `Bid_Number`, `Title`, `Due_Date`, `company_name` |
| Submission | `Bid_Submission_Type`, `Any_Additional_Documentation_Required`, `Bid_Bond_Requirement`, `MFG_for_Registration` |
| Timeline & terms | `Pre_Bid_Meeting`, `Term_of_Bid`, `Delivery_Date`, `Payment_Terms` |
| Technical | `Product`, `Product_Specification`, `Model_no`, `Part_no`, `Installation` |
| Contact & summary | `contact_info`, `Contract_or_Cooperative_to_use`, `Bid_Summary` |
| Extra | `Additional_Information` – evaluation criteria, full timeline, vendor qualifications, compliance, warranty, special conditions, procurement type, contract terms |

## Project structure

```
rfp-information-extractor/
├── rfp_extractor.py        # the full pipeline + CLI
├── data/
│   ├── Bid1/               # Student & Staff Computing Devices (RFP + 2 addenda + bid page)
│   └── Bid2/               # Dell Laptops w/ Extended Warranty (RFP, specs, affidavits, bid page)
├── output/
│   └── all_extractions.json  # generated results
├── tests/
│   └── test_extractor.py   # offline unit tests (no API key needed)
├── requirements.txt
├── .env.example
└── LICENSE
```

## Getting started

**Requirements:** Python 3.9+ and an [Anthropic API key](https://console.anthropic.com/).

```bash
git clone https://github.com/CarolDsillva/rfp-information-extractor.git
cd rfp-information-extractor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then paste your key into .env
```

You can also set the key as an environment variable (`export ANTHROPIC_API_KEY=...`, or `$env:ANTHROPIC_API_KEY="..."` in PowerShell) or pass `--api-key`.

## Usage

Run on the included sample bids:

```bash
python rfp_extractor.py data/Bid1 data/Bid2
```

Results are written to `output/all_extractions.json`, keyed by `<folder>/<file>`.

```bash
# Single file or any mix of files and folders
python rfp_extractor.py path/to/rfp.pdf
python rfp_extractor.py urgent.pdf data/Bid1 data/Bid2

# Custom output path, plus one JSON file per document
python rfp_extractor.py data/Bid1 -o results/bid1.json --individual-files

# Use a different Claude model
python rfp_extractor.py data/Bid1 --model claude-sonnet-4-5
```

| Option | Description |
|---|---|
| `paths` | One or more files or directories (`.pdf`, `.html`, `.htm`) |
| `-o, --output` | Combined output file (default `output/all_extractions.json`) |
| `--individual-files` | Also save one JSON per document next to the combined file |
| `--model` | Claude model ID (default `claude-sonnet-4-5`, or `RFP_EXTRACTOR_MODEL` env var) |
| `--api-key` | API key, if not set in the environment / `.env` |

### Sample output (abridged)

```json
{
  "Bid1/JA-207652 Student and Staff Computing Devices FINAL.pdf": {
    "Bid_Number": "JA-207652",
    "Title": "Student and Staff Computing Devices",
    "Due_Date": "...",
    "Bid_Submission_Type": "...",
    "Product": "...",
    "...": "...",
    "Additional_Information": "Evaluation Criteria: ...; Timeline: ...; Warranty: ..."
  },
  "Bid2/PORFP_-_Dell_Laptop_Final.pdf": { "...": "..." }
}
```

See [`output/all_extractions.json`](output/all_extractions.json) for the full results on the sample data.

## Tests

The tests cover JSON parsing, schema normalisation, chunking and file parsing, and run without an API key:

```bash
pip install pytest
pytest
```

## Design notes & limitations

- **Per-document output.** Each file is extracted independently, as the brief asked. Merging a main RFP with its addenda into one consolidated record per bid is a natural next step.
- **Text-based PDFs only.** Scanned PDFs return no text; adding OCR (e.g. Tesseract) would cover them.
- **Field-group retrieval** trades a few extra embedding calls for much better recall on long documents than a single "summarise this RFP" query.
- **Cost:** one Claude call per document. Small documents skip retrieval entirely, and the embedding model only loads when a large document appears.

## Sample data

The documents in `data/` are publicly posted procurement documents (via BidNet Direct) that were provided as part of the assignment. They're included so the project can be run end-to-end.

## License

[MIT](LICENSE) © Carol D'Silva
