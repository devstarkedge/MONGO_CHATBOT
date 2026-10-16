"""Split long text into chunks that fit the embedding model's window (~256 tokens)."""


def split_long_line(line: str, size: int, overlap: int) -> list:
    """Cut one very long line into overlapping pieces, preferring to break at spaces."""
    pieces, start, n = [], 0, len(line)
    while start < n:
        end = min(start + size, n)
        if end < n:
            space = line.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        piece = line[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_lines(lines: list, size: int, overlap: int) -> list:
    """Pack 'Label: value' lines into chunks of at most `size` characters."""
    chunks, current, length = [], [], 0
    for line in lines:
        parts = [line] if len(line) <= size else split_long_line(line, size, overlap)
        for part in parts:
            if current and length + len(part) + 1 > size:
                chunks.append("\n".join(current))
                current, length = [], 0
            current.append(part)
            length += len(part) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks
