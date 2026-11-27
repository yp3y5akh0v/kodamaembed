"""Create deterministic, exact-length fixtures of a model family from a caller-supplied corpus."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from _matched import FAMILIES

# A sequence fixture holds a text of every length from here to the family's longest.
SEQUENCE_FIRST = 16
SEQUENCE_STEP, SEQUENCE_OFFSET = 83, 27
CONTINUED = " Each document is continued by those after it."


def sequence_order(count):
    """Text indices in call order, each once; consecutive calls differ in length."""
    if math.gcd(SEQUENCE_STEP, count) != 1:
        raise ValueError(f"the order's step shares a factor with {count} texts")
    return [(SEQUENCE_STEP * index + SEQUENCE_OFFSET) % count for index in range(count)]


def first_prefix(documents, tokenizer, prefix, length):
    for document in documents:
        words = document.get("text", "").split()
        low, high = 1, len(words)
        # Encoded length never falls as words are added, so bisection finds a match.
        while low <= high:
            count = (low + high) // 2
            encoded = tokenizer.encode(prefix + " ".join(words[:count]), add_special_tokens=True)
            if len(encoded.ids) < length:
                low = count + 1
            elif len(encoded.ids) > length:
                high = count - 1
            else:
                return document["_id"], " ".join(words[:count]), encoded
    raise ValueError(f"corpus has no word prefix of {length} tokens")


def running_texts(documents, words):
    """Each document continued by those after it, to `words` words at most: a text as long as a case
    needs where no single document is."""
    for index, document in enumerate(documents):
        text = []
        for following in documents[index:]:
            text += following.get("text", "").split()
            if len(text) >= words:
                break
        yield {"_id": document["_id"], "text": " ".join(text[:words])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--family", choices=tuple(FAMILIES), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-corpus-sha256")
    parser.add_argument("--running-text", action="store_true",
                        help="continue each document with those after it, for a family whose longest cases no "
                             "single document reaches")
    parser.add_argument("--sequence", action="store_true",
                        help="also write <family>-sequence.json for run_length_sequence.py: a text of every "
                             f"length from {SEQUENCE_FIRST} tokens to the family's longest, and their call order")
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("--out-dir must be new or empty")
    from tokenizers import Tokenizer

    corpus = args.corpus.read_bytes()
    corpus_hash = hashlib.sha256(corpus).hexdigest()
    if args.expected_corpus_sha256 and args.expected_corpus_sha256 != corpus_hash:
        parser.error("corpus hash differs from the declared workload")
    tokenizer_bytes = (args.model_dir / "tokenizer.json").read_bytes()
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    family = FAMILIES[args.family]
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=family.max_length, stride=0, strategy="longest_first", direction="right")
    documents = [json.loads(line) for line in corpus.decode("utf-8").splitlines()]
    prefix = family.query_prefix
    continued = CONTINUED if args.running_text else ""

    def select(length):
        # A word is at least one token, so a prefix of `length` tokens has no more words.
        candidates = running_texts(documents, length) if args.running_text else documents
        return first_prefix(candidates, tokenizer, prefix, length)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for length in family.lengths:
        document, text, encoded = select(length)
        label = f"{args.family}-b1-l{length}"
        fixture = {
            "label": label, "family": args.family, "role": "query",
            "texts": [text], "expected_lengths": [length],
            "expected_ids": [encoded.ids], "expected_types": [encoded.type_ids],
            "source_document_ids": [document],
            "source_corpus_sha256": corpus_hash,
            "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
            "selection": "First corpus word prefix with the exact encoded length under the model's truncation."
                         + continued,
        }
        with (args.out_dir / f"{label}.json").open("x", encoding="utf-8") as destination:
            json.dump(fixture, destination, indent=2, ensure_ascii=True)
            destination.write("\n")
        print(label)
    if args.sequence:
        lengths = list(range(SEQUENCE_FIRST, family.max_length + 1))
        selected = [select(length) for length in lengths]
        label = f"{args.family}-sequence"
        fixture = {
            "label": label, "family": args.family, "role": "query",
            "texts": [text for _, text, _ in selected], "expected_lengths": lengths,
            "expected_ids": [encoded.ids for _, _, encoded in selected],
            "expected_types": [encoded.type_ids for _, _, encoded in selected],
            "order": sequence_order(len(lengths)),
            "order_rule": f"({SEQUENCE_STEP} i + {SEQUENCE_OFFSET}) mod {len(lengths)} over the texts, "
                          "which stand in ascending length",
            "source_document_ids": [document for document, _, _ in selected],
            "source_corpus_sha256": corpus_hash,
            "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
            "selection": "For each length, the first corpus word prefix with the exact encoded length under the "
                         "model's truncation." + continued,
        }
        with (args.out_dir / f"{label}.json").open("x", encoding="utf-8") as destination:
            json.dump(fixture, destination, indent=2, ensure_ascii=True)
            destination.write("\n")
        print(label)


if __name__ == "__main__":
    main()
