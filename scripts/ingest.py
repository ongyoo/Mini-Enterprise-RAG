import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app.config import get_settings
from app.db.session import build_engine, build_session_factory
from app.logging_config import configure_logging
from app.rag.embeddings import build_embeddings
from app.rag.ingest import ingest_file, load_documents


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Idempotent ingestion of the synthetic corpus")
    parser.add_argument("--path", type=Path, default=settings.documents_path)
    parser.add_argument("--check", action="store_true", help="validate the file only")
    args = parser.parse_args()

    if args.check:
        docs, errors = load_documents(args.path)
        print(f"{len(docs)} valid documents, {len(errors)} errors")
        for error in errors:
            print(f"  {error}")
        return 1 if errors or not docs else 0

    configure_logging(settings.log_level)
    engine = build_engine(settings.database_url)
    report = ingest_file(
        args.path, build_session_factory(engine), build_embeddings(settings), settings
    )
    print(json.dumps(asdict(report) | {"status": report.status}, ensure_ascii=False, indent=2))
    return 0 if report.status == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
