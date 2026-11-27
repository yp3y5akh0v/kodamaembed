"""Model names the suite uses: two small enough for any machine's processor,
the quantized preset's model, and one KodamaEmbed does not load."""

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
E5 = "intfloat/e5-large"
UNREGISTERED = "sentence-transformers/all-distilroberta-v1"
