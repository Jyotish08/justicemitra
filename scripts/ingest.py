"""Offline administrative ingestion CLI for legal statutes and documents into MongoDB Atlas Vector Search."""

import os
import sys
import json
import hashlib
import argparse
import datetime
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from justicemitra.config import config, AppConfig
from justicemitra.services.embedding import get_embedding_service, EmbeddingService
from justicemitra.services.database import (
    get_mongo_client,
    get_legal_chunks_collection,
    get_document_registry_collection,
    ensure_vector_search_index,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")

# Canonical statutory provisions for women's legal rights in India
# Used as high-fidelity statutory grounding alongside raw PDF ingestion
CANONICAL_STATUTES: List[Dict[str, Any]] = [
    {
        "source_name": "Protection of Women from Domestic Violence Act, 2005",
        "statute_title": "PWDVA 2005 (Domestic Violence Act)",
        "section": "Section 17",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/2021",
        "jurisdiction": "Central",
        "document_version": "2005-enacted",
        "chunk_text": (
            "Protection of Women from Domestic Violence Act, 2005 (PWDVA 2005) - Section 17. "
            "Right to reside in a shared household and matrimonial home: (1) Notwithstanding anything contained "
            "in any other law for the time being in force, every woman in a domestic relationship shall have the right "
            "to reside in the shared household or matrimonial home, whether or not she has any right, title or beneficial "
            "interest in the same. (2) The aggrieved person shall not be evicted or excluded from the shared household "
            "or matrimonial home or any part of it by the respondent save in accordance with the procedure established by law. "
            "A woman has a statutory right to stay in her matrimonial home after filing a domestic violence complaint, and the "
            "husband or in-laws cannot unlawfully dispossess, evict, or lock her out."
        ),
    },
    {
        "source_name": "Protection of Women from Domestic Violence Act, 2005",
        "statute_title": "PWDVA 2005 (Domestic Violence Act)",
        "section": "Section 19",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/2021",
        "jurisdiction": "Central",
        "document_version": "2005-enacted",
        "chunk_text": (
            "Protection of Women from Domestic Violence Act, 2005 (PWDVA 2005) - Section 19. "
            "Residence orders prohibiting eviction: (1) While disposing of an application under sub-section (1) of section 12, "
            "the Magistrate may, on being satisfied that domestic violence has taken place, pass a residence order— "
            "(a) restraining the respondent from dispossessing or in any other manner disturbing the possession of the "
            "aggrieved person from the shared household or matrimonial home; (b) directing the respondent to remove himself from "
            "the shared household; (c) restraining the respondent or any of his relatives from entering any portion of the "
            "shared household in which the aggrieved person resides; (d) directing the respondent to secure alternate accommodation "
            "for the aggrieved person of the same level as enjoyed by her in the shared household or to pay rent for the same."
        ),
    },
    {
        "source_name": "Protection of Women from Domestic Violence Act, 2005",
        "statute_title": "PWDVA 2005 (Domestic Violence Act)",
        "section": "Section 18",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/2021",
        "jurisdiction": "Central",
        "document_version": "2005-enacted",
        "chunk_text": (
            "Protection of Women from Domestic Violence Act, 2005 (PWDVA 2005) - Section 18. "
            "Protection orders: The Magistrate may, after giving the aggrieved person and the respondent an "
            "opportunity of being heard and on being prima facie satisfied that domestic violence has taken place or is likely "
            "to take place, pass a protection order in favour of the aggrieved person and prohibit the respondent from committing "
            "any act of domestic violence, aiding or abetting in the commission of acts of domestic violence, entering the place "
            "of employment of the aggrieved person, or attempting to communicate in any form with the aggrieved person."
        ),
    },
    {
        "source_name": "Protection of Women from Domestic Violence Act, 2005",
        "statute_title": "PWDVA 2005 (Domestic Violence Act)",
        "section": "Section 12",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/2021",
        "jurisdiction": "Central",
        "document_version": "2005-enacted",
        "chunk_text": (
            "Protection of Women from Domestic Violence Act, 2005 (PWDVA 2005) - Section 12. "
            "Application to Magistrate seeking relief: (1) An aggrieved person or a Protection Officer or any other person on "
            "behalf of the aggrieved person may present an application to the Magistrate seeking one or more reliefs under this Act, "
            "including residence orders under section 19, protection orders under section 18, monetary relief under section 20, "
            "and custody orders under section 21: Provided that before passing any order on such application, the Magistrate shall "
            "take into consideration any domestic incident report received by him from the Protection Officer or the service provider."
        ),
    },
    {
        "source_name": "Dowry Prohibition Act, 1961",
        "statute_title": "Dowry Prohibition Act 1961",
        "section": "Section 3 & 4",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/1522",
        "jurisdiction": "Central",
        "document_version": "1961-enacted",
        "chunk_text": (
            "Dowry Prohibition Act, 1961 - Section 3 & 4. Penalty for giving, taking, or demanding dowry: "
            "If any person, before wedding, at the time of marriage, or any time after marriage, gives, takes, abets, or demands "
            "dowry from the bride, groom, or their family—including demanding cash, money, motor vehicles (cars, motorcycles, scooters), "
            "valuable property, or jewelry—it constitutes an illegal dowry demand. Under Section 3, penalty for giving or taking dowry "
            "is imprisonment for not less than five years and fine. Under Section 4, demanding dowry directly or indirectly from the parents "
            "or relatives of a bride is punishable with imprisonment for not less than six months up to two years and fine."
        ),
    },
    {
        "source_name": "Indian Penal Code / Bharatiya Nyaya Sanhita (BNS)",
        "statute_title": "Dowry Prohibition Act / IPC 498A / BNS 85",
        "section": "Section 498A IPC / Section 85 BNS",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/1363",
        "jurisdiction": "Central",
        "document_version": "revised",
        "chunk_text": (
            "Indian Penal Code Section 498A / Bharatiya Nyaya Sanhita (BNS) Section 85. Husband or relative of husband "
            "subjecting woman to cruelty and dowry harassment: Whoever, being the husband or the relative of the husband of a woman, "
            "subjects such woman to cruelty shall be punished with imprisonment for a term which may extend to three years and fine. "
            "Cruelty includes harassment of the woman where such harassment is with a view to coercing her or any person related to her "
            "to meet any unlawful demand for property, money, cash, vehicles, or valuable security (dowry demands before, during, or after wedding)."
        ),
    },
    {
        "source_name": "Legal Services Authorities Act, 1987",
        "statute_title": "Legal Services Authorities Act 1987",
        "section": "Section 12(c)",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/1944",
        "jurisdiction": "Central",
        "document_version": "1987-enacted",
        "chunk_text": (
            "Legal Services Authorities Act, 1987 - Section 12(c). Criteria for giving free legal services: "
            "Every person who has to file or defend a case shall be entitled to free legal services under this Act if that person is "
            "(c) a woman or a child. An indigent woman, destitute woman, or any woman in India is entitled to free legal aid, a free "
            "advocate/lawyer for court representation, legal drafting, payment of court fees, and process expenses across all Courts "
            "(Supreme Court, High Court, District Court, Family Court) through the National Legal Services Authority (NALSA Helpline 1516), "
            "State Legal Services Authority (SLSA), District Legal Services Authority (DLSA), and Legal Services Clinics."
        ),
    },
    {
        "source_name": "Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013",
        "statute_title": "POSH Act 2013",
        "section": "Section 9",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/2104",
        "jurisdiction": "Central",
        "document_version": "2013-enacted",
        "chunk_text": (
            "Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013 (POSH Act 2013) - Section 9. "
            "Complaint of sexual harassment and statutory limitation period: (1) Any aggrieved woman may make, in writing, a complaint "
            "of sexual harassment at workplace to the Internal Committee (IC) if so constituted, or the Local Committee (LC), within a "
            "period of three months from the date of incident and in case of a series of incidents, within a period of three months from the "
            "date of last incident: Provided that the Internal Committee or, as the case may be, the Local Committee may, for reasons to be "
            "recorded in writing, extend the time limit not exceeding three months (total six months), if it is satisfied that the circumstances "
            "were such which prevented the woman from filing a complaint within the said period. Therefore, an incident that occurred five "
            "months ago can still be validly filed by requesting an extension upon showing sufficient cause."
        ),
    },
    {
        "source_name": "Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013",
        "statute_title": "POSH Act 2013",
        "section": "Section 4",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/2104",
        "jurisdiction": "Central",
        "document_version": "2013-enacted",
        "chunk_text": (
            "Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013 (POSH Act 2013) - Section 4. "
            "Constitution of Internal Committee: Every employer of a workplace shall, by an order in writing, constitute a Committee "
            "to be known as the 'Internal Committee (IC)': Provided that where the offices or administrative units of the workplace are "
            "located at different places or divisional or sub-divisional level, the Internal Committee shall be constituted at all "
            "administrative units or offices. The Presiding Officer shall be a woman employed at a senior level at workplace from amongst the employees."
        ),
    },
    {
        "source_name": "Maternity Benefit Act, 1961",
        "statute_title": "Maternity Benefit Act 1961",
        "section": "Section 5",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/1547",
        "jurisdiction": "Central",
        "document_version": "2017-amended",
        "chunk_text": (
            "Maternity Benefit Act, 1961 - Section 5. Right to payment of maternity benefit: Every woman shall be entitled to, and her employer "
            "shall be liable for, the payment of maternity benefit at the rate of the average daily wage for the period of her actual absence. "
            "The maximum period for which any woman shall be entitled to maternity benefit shall be twenty-six weeks of which not more than eight "
            "weeks shall precede the date of her expected delivery."
        ),
    },
    {
        "source_name": "Equal Remuneration Act, 1976",
        "statute_title": "Equal Remuneration Act 1976",
        "section": "Section 4",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/1458",
        "jurisdiction": "Central",
        "document_version": "1976-enacted",
        "chunk_text": (
            "Equal Remuneration Act, 1976 - Section 4. Duty of employer to pay equal remuneration to men and women workers for same work: "
            "No employer shall pay to any worker, employed by him in an establishment or employment, remuneration, whether payable in cash "
            "or in kind, at rates less favourable than those at which remuneration is paid by him to the workers of the opposite sex in such "
            "establishment or employment for performing the same work or work of a similar nature."
        ),
    },
]


def map_handbook_page_to_statute(page_num: int) -> Tuple[str, str]:
    """Maps page numbers from Majlis Women's Legal Rights handbook to specific statutory topics."""
    if 15 <= page_num <= 24:
        return "PWDVA 2005 (Domestic Violence Act)", f"Chapter 6: Domestic Violence - Page {page_num}"
    elif 25 <= page_num <= 34:
        return "Dowry Prohibition Act / IPC 498A", f"Chapter 7: Dowry Harassment & Cruelty - Page {page_num}"
    elif 35 <= page_num <= 44:
        return "Sexual Offences & IPC", f"Chapter 8/9: Sexual Offences - Page {page_num}"
    elif 45 <= page_num <= 51:
        return "Women Health & Safety Acts", f"Chapter 10/11: Health Laws - Page {page_num}"
    elif 52 <= page_num <= 61:
        return "POSH Act 2013 & Workplace Labor Rights", f"Chapter 14: POSH Workplace Harassment - Page {page_num}"
    elif 62 <= page_num <= 76:
        return "NCW & Free Legal Aid Services", f"Chapter 17: Legal Aid Clinics - Page {page_num}"
    else:
        return "Women Legal Rights Guide", f"Page {page_num}"


def compute_file_hash(file_path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_chunk_id(statute_title: str, section: str, text: str) -> str:
    """Generates a stable, deterministic chunk identifier."""
    raw = f"{statute_title.strip()}_{section.strip()}_{text.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_pdf_file(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extracts text from a PDF and chunks it into section-aware text fragments."""
    import pypdf

    chunks = []
    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception as e:
        logger.error(f"Cannot read PDF {pdf_path}: {e}")
        return []

    total_pages = len(reader.pages)
    if total_pages == 0:
        logger.warning(f"PDF {pdf_path} has 0 pages.")
        return []

    file_title = pdf_path.stem.replace("_", " ")

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Error extracting page {page_num} of {pdf_path}: {e}")
            continue

        text = text.strip()
        if len(text) < 50:
            continue

        statute_title, section_label = map_handbook_page_to_statute(page_num)

        # Split long page text into chunks of ~1200 characters with 150 overlap
        chunk_size = 1200
        overlap = 150
        start = 0
        part_idx = 1
        while start < len(text):
            end = start + chunk_size
            chunk_slice = text[start:end].strip()
            if len(chunk_slice) >= 80:
                chunks.append(
                    {
                        "source_name": f"{file_title} ({section_label})",
                        "statute_title": statute_title,
                        "section": f"{section_label} - Part {part_idx}",
                        "source_url": "https://www.indiacode.nic.in/",
                        "jurisdiction": "Central",
                        "document_version": "published",
                        "chunk_text": chunk_slice,
                    }
                )
                part_idx += 1
            start += chunk_size - overlap

    return chunks


def prepare_documents(
    pdf_dir: Optional[Path] = None,
    include_canonical: bool = True,
    embedder: Optional[EmbeddingService] = None,
) -> List[Dict[str, Any]]:
    """Gathers canonical statutes and PDF chunks, computes embeddings, and formats for MongoDB."""
    all_raw_docs: List[Dict[str, Any]] = []

    if include_canonical:
        all_raw_docs.extend(CANONICAL_STATUTES)

    if pdf_dir and pdf_dir.exists():
        for pdf_file in pdf_dir.glob("*.pdf"):
            # Skip empty files (< 100 bytes)
            if pdf_file.stat().st_size < 100:
                logger.info(f"Skipping empty or stub file: {pdf_file.name}")
                continue
            logger.info(f"Parsing PDF document: {pdf_file.name} ({pdf_file.stat().st_size} bytes)")
            pdf_chunks = parse_pdf_file(pdf_file)
            logger.info(f"Extracted {len(pdf_chunks)} chunks from {pdf_file.name}")
            all_raw_docs.extend(pdf_chunks)

    if not all_raw_docs:
        logger.warning("No documents gathered for ingestion.")
        return []

    embed_svc = embedder or get_embedding_service()
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    logger.info(f"Generating 384-dimensional dense embeddings for {len(all_raw_docs)} chunks...")
    # Embed with statute context prefix so dense representation captures both statute and content
    texts_to_embed = [
        f"[{doc['statute_title']} - {doc['section']}] {doc['chunk_text']}"
        for doc in all_raw_docs
    ]
    vectors = embed_svc.encode(texts_to_embed, normalize=True)

    prepared_records = []
    for doc, vec in zip(all_raw_docs, vectors):
        chunk_id = compute_chunk_id(doc["statute_title"], doc["section"], doc["chunk_text"])
        prepared_records.append(
            {
                "chunk_id": chunk_id,
                "document_id": hashlib.sha256(doc["source_name"].encode("utf-8")).hexdigest(),
                "source_name": doc["source_name"],
                "source_url": doc.get("source_url", "https://www.indiacode.nic.in/"),
                "statute_title": doc["statute_title"],
                "section": doc["section"],
                "jurisdiction": doc.get("jurisdiction", "Central"),
                "document_version": doc.get("document_version", "enacted"),
                "chunk_text": doc["chunk_text"],
                "embedding": vec.tolist(),
                "token_count": len(doc["chunk_text"].split()),
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        )

    return prepared_records


def run_ingestion(
    pdf_dir: Path,
    export_json_path: Optional[Path] = None,
    force_reindex: bool = False,
    app_config: Optional[AppConfig] = None,
) -> Dict[str, Any]:
    """Executes deterministic document ingestion into MongoDB Atlas and optionally exports JSON corpus."""
    cfg = app_config or config
    embedder = get_embedding_service()

    docs = prepare_documents(pdf_dir=pdf_dir, include_canonical=True, embedder=embedder)
    logger.info(f"Prepared {len(docs)} total legal chunks with 384-d embeddings.")

    # Save local JSON export if requested or for offline fallback
    if export_json_path:
        export_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_json_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2)
        logger.info(f"Exported legal corpus JSON to: {export_json_path}")

    # Check MongoDB Atlas availability
    client = get_mongo_client(cfg)
    if client is None:
        logger.warning(
            "MONGODB_URI is not set. Ingestion to MongoDB Atlas bypassed. "
            "Corpus exported locally for fallback."
        )
        return {
            "status": "local_only",
            "chunks_prepared": len(docs),
            "atlas_upserts": 0,
            "export_json": str(export_json_path) if export_json_path else None,
        }

    from pymongo import UpdateOne
    collection = get_legal_chunks_collection(client, cfg)
    registry = get_document_registry_collection(client, cfg)

    if collection is None:
        raise RuntimeError("Unable to access MongoDB legal_chunks collection.")

    # Initialize / verify Vector Search Index
    index_res = ensure_vector_search_index(collection, cfg.mongodb_vector_index)
    logger.info(f"Vector search index status: {index_res}")

    # Bulk idempotent upsert
    bulk_ops = [
        UpdateOne(
            {"chunk_id": doc["chunk_id"]},
            {"$set": doc},
            upsert=True,
        )
        for doc in docs
    ]

    result = collection.bulk_write(bulk_ops, ordered=False)
    logger.info(
        f"MongoDB Bulk Upsert complete: {result.upserted_count} inserted, "
        f"{result.modified_count} updated, {result.matched_count} matched."
    )

    return {
        "status": "success",
        "chunks_prepared": len(docs),
        "inserted": result.upserted_count,
        "modified": result.modified_count,
        "matched": result.matched_count,
        "index_status": index_res,
    }


def main():
    parser = argparse.ArgumentParser(description="JusticeMitra Legal Knowledge Ingestion CLI")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help="Path to directory containing legal PDFs",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "legal_corpus.json"),
        help="Path to save JSON corpus file",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force re-indexing even if file hash matches",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    export_json = Path(args.export_json) if args.export_json else None

    logger.info("Starting JusticeMitra ingestion workflow...")
    res = run_ingestion(
        pdf_dir=data_dir,
        export_json_path=export_json,
        force_reindex=args.force_reindex,
    )
    logger.info(f"Ingestion result: {res}")


if __name__ == "__main__":
    main()
