#!/usr/bin/env python3
"""
RFP Information Extractor
=========================

Extracts structured bid information from RFP documents (PDF and HTML) using
Claude and a lightweight retrieval-augmented generation (RAG) step for large
documents.

Pipeline:
    1. Parse   - pull raw text out of PDF (pypdf) or HTML (BeautifulSoup)
    2. Retrieve - for large documents, chunk + embed the text and keep only the
                  chunks most relevant to the target fields
    3. Extract  - ask Claude to map the text onto a fixed 21-field schema
    4. Output   - validate the JSON against the schema and write it to disk

Author: Carol D'Silva
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================================
# Configuration
# ============================================================================

SUPPORTED_FORMATS = {".pdf", ".html", ".htm"}
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_MODEL = os.getenv("RFP_EXTRACTOR_MODEL", "claude-sonnet-4-5")
DEFAULT_OUTPUT = "output/all_extractions.json"

CHUNK_SIZE = 400            # words per chunk
CHUNK_OVERLAP = 80          # words shared between neighbouring chunks
CHUNKS_PER_QUERY = 3        # top-k chunks kept per retrieval query
LARGE_DOC_THRESHOLD = 50_000  # characters; above this, RAG is used
RETRIEVAL_BUDGET_CHARS = 60_000  # max characters of retrieved context
MAX_PROMPT_CHARS = 100_000  # hard cap on document text sent to the model
MAX_OUTPUT_TOKENS = 8_000

# One retrieval query per group of related fields. Querying per field group
# (rather than one generic query) makes sure details buried deep in a long RFP,
# such as bond requirements or payment terms, still reach the model.
RETRIEVAL_QUERIES = [
    "solicitation number RFP bid number title of the request for proposal",
    "proposal due date deadline submission time timezone",
    "how to submit proposals electronic portal email sealed physical delivery",
    "contract term duration renewal options",
    "pre-bid pre-proposal conference meeting date location virtual",
    "installation deployment setup imaging requirements",
    "bid bond performance bond surety percentage amount",
    "delivery date timeline shipping lead time",
    "payment terms net days invoice",
    "required forms documents certifications affidavits attachments",
    "manufacturer authorized reseller registration OEM",
    "cooperative purchasing piggyback contract interlocal agreement",
    "model number part number SKU product specifications quantity",
    "contact person procurement officer email phone address",
    "evaluation criteria scoring points weighting",
    "warranty requirements service repair",
    "vendor qualifications references experience",
    "compliance federal state requirements MWBE HUB set-aside",
]


# ============================================================================
# Data Model
# ============================================================================

@dataclass
class RFPData:
    """Target schema for extracted RFP information (21 fields)."""

    Bid_Number: str = ""
    Title: str = ""
    Due_Date: str = ""
    Bid_Submission_Type: str = ""
    Term_of_Bid: str = ""
    Pre_Bid_Meeting: str = ""
    Installation: str = ""
    Bid_Bond_Requirement: str = ""
    Delivery_Date: str = ""
    Payment_Terms: str = ""
    Any_Additional_Documentation_Required: str = ""
    MFG_for_Registration: str = ""
    Contract_or_Cooperative_to_use: str = ""
    Model_no: str = ""
    Part_no: str = ""
    Product: str = ""
    contact_info: str = ""
    company_name: str = ""
    Bid_Summary: str = ""
    Product_Specification: str = ""
    Additional_Information: str = ""

    @classmethod
    def field_names(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def from_dict(cls, data: Dict) -> "RFPData":
        """Build a record from model output, keeping only known fields and
        coercing every value to a string so the output schema is stable."""
        clean = {}
        for name in cls.field_names():
            value = data.get(name, "")
            if value is None:
                value = ""
            elif isinstance(value, list):
                value = "; ".join(str(v) for v in value)
            elif isinstance(value, dict):
                value = "; ".join(f"{k}: {v}" for k, v in value.items())
            clean[name] = str(value).strip()
        return cls(**clean)


# ============================================================================
# Document Parsing
# ============================================================================

class DocumentParser:
    """Extracts plain text from PDF and HTML files."""

    @staticmethod
    def parse_pdf(filepath: Path) -> str:
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(filepath))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:  # corrupt or encrypted PDF
            print(f"  ! Could not read PDF {filepath.name}: {exc}")
            return ""
        return "\n".join(p for p in pages if p.strip())

    @staticmethod
    def parse_html(filepath: Path) -> str:
        from bs4 import BeautifulSoup

        try:
            html = filepath.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            print(f"  ! Could not read HTML {filepath.name}: {exc}")
            return ""

        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style", "noscript", "svg", "head"]):
            element.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Collapse runs of blank lines left behind by layout markup
        return re.sub(r"\n{3,}", "\n\n", text)

    def parse(self, filepath: Path) -> str:
        ext = filepath.suffix.lower()
        if ext == ".pdf":
            return self.parse_pdf(filepath)
        if ext in {".html", ".htm"}:
            return self.parse_html(filepath)
        raise ValueError(f"Unsupported format: {ext}")


# ============================================================================
# Retrieval (RAG)
# ============================================================================

def chunk_text(text: str, size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word windows."""
    if size <= overlap:
        raise ValueError("chunk size must be larger than overlap")
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)
            if words[i:i + size]]


class SemanticSearchEngine:
    """Embeds document chunks and retrieves the ones most relevant to a set of
    field-specific queries. The embedding model is loaded lazily, so small
    documents never pay the start-up cost."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def retrieve(self, text: str, queries: List[str],
                 top_k: int = CHUNKS_PER_QUERY,
                 budget_chars: int = RETRIEVAL_BUDGET_CHARS) -> str:
        import numpy as np

        chunks = chunk_text(text)
        if not chunks:
            return ""

        # Normalised embeddings -> dot product == cosine similarity
        chunk_emb = self.model.encode(chunks, convert_to_numpy=True,
                                      normalize_embeddings=True)
        query_emb = self.model.encode(queries, convert_to_numpy=True,
                                      normalize_embeddings=True)
        scores = query_emb @ chunk_emb.T  # (n_queries, n_chunks)

        # Candidates: the top-k chunks for every query, ranked by their best
        # score against any query. The first chunk is always kept because it
        # nearly always carries the title, bid number and key dates.
        best = scores.max(axis=0)
        candidates = {0}
        for row in scores:
            candidates.update(np.argsort(row)[-top_k:].tolist())
        ranked = sorted(candidates, key=lambda i: (i != 0, -best[i]))

        # Fill a character budget with the most relevant chunks
        selected, used = [], 0
        for i in ranked:
            if used + len(chunks[i]) > budget_chars and selected:
                continue
            selected.append(i)
            used += len(chunks[i])

        # Keep document order so the model reads a coherent excerpt
        return "\n\n[...]\n\n".join(chunks[i] for i in sorted(selected))


# ============================================================================
# LLM Extraction
# ============================================================================

SYSTEM_PROMPT = (
    "You are a procurement analyst who extracts bid details from RFP "
    "documents. You only report information that is present in the document "
    "and you always answer with a single valid JSON object."
)

FIELD_GUIDE = """\
1. Bid_Number - Bid / RFP / solicitation number
2. Title - Document title
3. Due_Date - Submission deadline with time and timezone
4. Bid_Submission_Type - How to submit (portal, email, physical)
5. Term_of_Bid - Contract duration and renewal options
6. Pre_Bid_Meeting - Meeting date, time, location, and virtual info
7. Installation - Deployment and installation requirements
8. Bid_Bond_Requirement - Bond amount or percentage
9. Delivery_Date - Expected delivery timeline
10. Payment_Terms - Payment conditions and timing
11. Any_Additional_Documentation_Required - Required forms and documents
12. MFG_for_Registration - Manufacturer / authorized reseller requirements
13. Contract_or_Cooperative_to_use - Piggyback or cooperative contracts
14. Model_no - Model numbers
15. Part_no - Part numbers
16. Product - Products or services requested
17. contact_info - Contact details
18. company_name - Issuing organization
19. Bid_Summary - 2-3 sentence summary
20. Product_Specification - Technical specifications
21. Additional_Information - Formatted as "Category: details; Category: details"
    using these categories when applicable: Evaluation Criteria (with point
    values), Timeline (all milestone dates), Vendor Qualifications,
    Compliance (Federal / State / Industry), Warranty, Submission
    Requirements, Special Conditions, Procurement Type, Contract Terms"""


def build_prompt(text: str) -> str:
    template = {name: "" for name in RFPData.field_names()}
    return f"""Extract structured information from the RFP document below.

<document>
{text[:MAX_PROMPT_CHARS]}
</document>

Guidelines:
- Extract information exactly as stated; do not guess.
- Leave a field as an empty string if the document does not contain it.
- Preserve date formats from the document.
- Separate multiple items with semicolons.
- Every value must be a string.

Fields:
{FIELD_GUIDE}

Return ONLY a JSON object with exactly these keys (no markdown, no commentary):
{json.dumps(template, indent=2)}"""


def parse_model_json(response_text: str) -> Dict:
    """Pull the JSON object out of a model response, tolerating markdown
    fences or stray text around it."""
    cleaned = response_text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


class RFPExtractor:
    """Runs parse -> retrieve -> extract for a single document."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key, max_retries=3)
        self.model = model
        self.parser = DocumentParser()
        self.search_engine = SemanticSearchEngine()

    def process_document(self, filepath: Path) -> Dict[str, str]:
        print(f"\nProcessing: {filepath}")
        text = self.parser.parse(filepath)

        if not text.strip():
            print("  ! No text extracted (scanned or empty document?)")
            return asdict(RFPData())

        if len(text) > LARGE_DOC_THRESHOLD:
            print(f"  - Large document ({len(text):,} chars): "
                  "retrieving relevant sections")
            text = self.search_engine.retrieve(text, RETRIEVAL_QUERIES)
            print(f"  - Reduced to {len(text):,} chars")

        return self._extract_with_llm(text, filepath)

    def _extract_with_llm(self, text: str, filepath: Path) -> Dict[str, str]:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_prompt(text)}],
            )
            response_text = "".join(
                block.text for block in response.content
                if getattr(block, "type", "") == "text"
            )
            if response.stop_reason == "max_tokens":
                print("  ! Response hit the token limit; output may be partial")
            return asdict(RFPData.from_dict(parse_model_json(response_text)))

        except json.JSONDecodeError as exc:
            print(f"  ! Model did not return valid JSON for "
                  f"{filepath.name}: {exc}")
        except Exception as exc:
            print(f"  ! Extraction failed for {filepath.name}: {exc}")
        return asdict(RFPData())


# ============================================================================
# Batch Processing
# ============================================================================

def collect_documents(paths: List[str]) -> List[tuple[Path, str]]:
    """Resolve CLI paths into (file, output_key) pairs.
    Files in a directory are keyed as "<dir>/<file>"; loose files by name."""
    documents = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: path not found: {path_str}")
        elif path.is_file():
            if path.suffix.lower() in SUPPORTED_FORMATS:
                documents.append((path, path.name))
            else:
                print(f"Warning: unsupported format: {path_str}")
        else:
            files = sorted(f for f in path.iterdir()
                           if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS)
            if not files:
                print(f"Warning: no supported files in {path_str}")
            documents.extend((f, f"{path.name}/{f.name}") for f in files)
    return documents


def safe_filename(key: str) -> str:
    stem = Path(key).with_suffix("").as_posix().replace("/", "_")
    return re.sub(r"[^\w.-]+", "_", stem).strip("_") + ".json"


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def process_paths(paths: List[str], output_file: str, api_key: str,
                  model: str = DEFAULT_MODEL,
                  save_individual: bool = False) -> Dict[str, Dict]:
    documents = collect_documents(paths)
    if not documents:
        print("No documents to process.")
        return {}

    extractor = RFPExtractor(api_key, model=model)
    output_path = Path(output_file)
    results: Dict[str, Dict] = {}
    failures = 0

    for filepath, key in documents:
        result = extractor.process_document(filepath)
        results[key] = result

        filled = sum(1 for v in result.values() if v)
        if filled == 0:
            failures += 1
        print(f"  ✓ {key}: {filled}/{len(result)} fields populated")

        if save_individual:
            individual = output_path.parent / safe_filename(key)
            write_json(result, individual)
            print(f"  ✓ Saved {individual}")

    write_json(results, output_path)

    print("\n" + "=" * 60)
    print(f"✓ Processed {len(results)} document(s)"
          + (f" ({failures} returned no data)" if failures else ""))
    print(f"✓ Results saved to {output_path}")
    print("=" * 60)
    return results


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured information from RFP documents "
                    "(PDF / HTML files or directories).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python rfp_extractor.py data/Bid1 data/Bid2
  python rfp_extractor.py document.pdf -o results.json
  python rfp_extractor.py data/Bid1 --individual-files
""",
    )
    parser.add_argument("paths", nargs="+",
                        help="files or directories containing RFP documents")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help=f"combined output file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Claude model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--api-key",
                        help="Anthropic API key (defaults to ANTHROPIC_API_KEY)")
    parser.add_argument("--individual-files", action="store_true",
                        help="also write one JSON file per document")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_arguments(argv)

    try:  # optional: load ANTHROPIC_API_KEY from a local .env file
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Anthropic API key required.\n"
              "Set ANTHROPIC_API_KEY (or add it to .env) or pass --api-key.")
        return 1

    try:
        process_paths(args.paths, args.output, api_key, model=args.model,
                      save_individual=args.individual_files)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
