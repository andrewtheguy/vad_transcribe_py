"""BinTokenizer format loader and token decoder."""


def load_tokenizer(path: str) -> list[bytes]:
    """Load a BinTokenizer .bin file, returning list mapping token ID to bytes."""
    vocab: list[bytes] = []
    with open(path, "rb") as f:
        while True:
            b = f.read(1)
            if not b:
                break
            first = b[0]
            if first == 0:
                vocab.append(b"")
                continue
            if first < 128:
                length = first
            else:
                second = f.read(1)
                if not second:
                    break
                length = (second[0] * 128) + first - 128
            token_bytes = f.read(length)
            if len(token_bytes) != length:
                raise ValueError(
                    f"Truncated tokenizer file {path}: token {len(vocab)} "
                    f"expected {length} bytes, got {len(token_bytes)}"
                )
            vocab.append(token_bytes)
    return vocab


def decode_tokens(token_ids: list[int], vocab: list[bytes]) -> str:
    """Decode token IDs to text."""
    parts: list[bytes] = []
    for tid in token_ids:
        if tid < 0 or tid >= len(vocab):
            continue
        tb = vocab[tid]
        # Skip special tokens like <pad>, <s>, </s>, <unk>
        if len(tb) >= 2 and tb[0:1] == b"<" and tb[-1:] == b">":
            continue
        parts.append(tb)
    text = b"".join(parts).decode("utf-8", errors="replace")
    # The tokenizer uses U+2581 (LOWER ONE EIGHTH BLOCK) as word separator
    text = text.replace("\u2581", " ").strip()
    return text
