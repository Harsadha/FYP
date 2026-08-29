def chunk_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs if paragraphs else [text.strip()]