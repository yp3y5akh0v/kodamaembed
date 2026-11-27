"""Write the first passages of a BEIR dataset with the test judgments that reach them and the
queries those judgments name, so a quality run on a large dataset has a bounded length."""

import argparse
import json
from pathlib import Path


def subset(source, target, passages):
    """Copies the selection from `source` into the new directory `target`, every kept line byte
    for byte; returns the counts of passages, queries and judgments."""
    with (source / "corpus.jsonl").open("rb") as corpus:
        kept = [line for _, line in zip(range(passages), corpus)]
    if len(kept) < passages:
        raise ValueError(f"the corpus holds only {len(kept)} passages")
    ids = {json.loads(line)["_id"] for line in kept}
    header, *rows = (source / "test.tsv").read_bytes().splitlines(keepends=True)
    judged = [row for row in rows if row.split(b"\t")[1].decode("utf-8") in ids]
    named = {row.split(b"\t")[0].decode("utf-8") for row in judged}
    with (source / "queries.jsonl").open("rb") as lines:
        queries = [line for line in lines if json.loads(line)["_id"] in named]
    target.mkdir(parents=True)
    (target / "corpus.jsonl").write_bytes(b"".join(kept))
    (target / "queries.jsonl").write_bytes(b"".join(queries))
    (target / "test.tsv").write_bytes(b"".join([header, *judged]))
    return len(kept), len(queries), len(judged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="directory with corpus.jsonl, queries.jsonl and test.tsv")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--passages", type=int, required=True)
    args = parser.parse_args()
    if args.passages < 1:
        parser.error("--passages must be positive")
    if args.out.exists():
        parser.error("--out must be a new directory")
    print("{} passages, {} queries, {} judgments".format(*subset(args.source, args.out, args.passages)))


if __name__ == "__main__":
    main()
