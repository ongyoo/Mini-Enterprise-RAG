import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

_HEADING = re.compile(r"(?m)^(?=#{1,3} )")


def split_content(
    *, title: str, content: str, doc_type: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """SOPs: one chunk per markdown heading, oversized sections split further.
    Chats: kept whole. Every chunk is prefixed with the title for retrieval context."""
    if doc_type == "Chat":
        return [f"{title}\n{content.strip()}"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks: list[str] = []
    for section in _HEADING.split(content):
        section = section.strip()
        if not section or not section.startswith("#"):
            continue
        for piece in splitter.split_text(section):
            chunks.append(f"{title}\n{piece}")
    return chunks or [f"{title}\n{content.strip()}"]
