//! Models by name, each pinned to one commit and to the hashes of its files.

use crate::{Error, Hub, ModelSpec, Pooling};
use std::{
    path::{Path, PathBuf},
    str::FromStr,
};

/// Why a file cannot be used as the pinned one, and how to replace it.
pub(crate) fn mismatch(path: &Path, actual: &str, pinned: &str) -> String {
    format!(
        "{} has SHA-256 {actual}, not the pinned {pinned}; delete it to download it again",
        path.display()
    )
}

/// The entry named `name` in any letter case, as Hugging Face resolves repository names.
fn parse<T: Copy>(
    kind: &str,
    name: &str,
    all: &[T],
    repository: impl Fn(T) -> &'static str,
) -> Result<T, Error> {
    all.iter()
        .copied()
        .find(|&model| repository(model).eq_ignore_ascii_case(name))
        .ok_or_else(|| {
            let known: Vec<&str> = all.iter().map(|&model| repository(model)).collect();
            Error::Unsupported(format!(
                "no {kind} named {name}; KodamaEmbed loads {}",
                known.join(", ")
            ))
        })
}

#[derive(Debug)]
pub(crate) struct Pinned {
    pub name: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

/// A repository at one commit; files in the order the loader reads them.
#[derive(Debug)]
pub(crate) struct Entry {
    pub repository: &'static str,
    pub revision: &'static str,
    pub files: [Pinned; 3],
}

impl Entry {
    /// Each file's SHA-256, in the loader's order.
    pub fn pins(&self) -> [&'static str; 3] {
        self.files.each_ref().map(|file| file.sha256)
    }

    pub fn download_size(&self) -> u64 {
        self.files.iter().map(|file| file.size).sum()
    }
}

/// Each file as (size in bytes, SHA-256), in the loader's order.
const fn files(
    config: (u64, &'static str),
    tokenizer: (u64, &'static str),
    weights: (u64, &'static str),
) -> [Pinned; 3] {
    [
        Pinned {
            name: "config.json",
            size: config.0,
            sha256: config.1,
        },
        Pinned {
            name: "tokenizer.json",
            size: tokenizer.0,
            sha256: tokenizer.1,
        },
        Pinned {
            name: "model.safetensors",
            size: weights.0,
            sha256: weights.1,
        },
    ]
}

struct EmbedderRecord {
    entry: Entry,
    spec: ModelSpec<'static>,
    dim: usize,
}

struct RerankerRecord {
    entry: Entry,
    max_length: usize,
}

/// BAAI's retrieval instruction: the BGE, mxbai and Arctic cards put it before queries only.
const SEARCH_QUERY: &str = "Represent this sentence for searching relevant passages: ";

/// The uncased BERT WordPiece tokenizer, byte for byte the same in BGE, E5 v2, mxbai and ms-marco.
const BERT_UNCASED_TOKENIZER: (u64, &str) = (
    711_396,
    "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
);

/// Both all-MiniLM repositories' export of that vocabulary; loading replaces its settings.
const ALL_MINILM_TOKENIZER: (u64, &str) = (
    466_247,
    "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
);

/// The tokenizer the base and large multilingual-e5 repositories ship, byte for byte.
const MULTILINGUAL_E5_TOKENIZER: (u64, &str) = (
    17_082_660,
    "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626",
);

/// The tokenizer both paraphrase-multilingual repositories ship, byte for byte.
const PARAPHRASE_MULTILINGUAL_TOKENIZER: (u64, &str) = (
    9_081_518,
    "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8",
);

/// The tokenizer both Arctic repositories ship, byte for byte.
const ARCTIC_TOKENIZER: (u64, &str) = (
    711_649,
    "91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854",
);

/// The tokenizer both ModernBERT repositories ship, byte for byte.
const MODERNBERT_TOKENIZER: (u64, &str) = (
    3_583_228,
    "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
);

const fn search(dim: usize) -> (ModelSpec<'static>, usize) {
    (
        ModelSpec {
            passage_prefix: "",
            query_prefix: SEARCH_QUERY,
            max_length: 512,
            pooling: Pooling::Cls,
        },
        dim,
    )
}

const fn e5(dim: usize) -> (ModelSpec<'static>, usize) {
    (
        ModelSpec {
            passage_prefix: "passage: ",
            query_prefix: "query: ",
            max_length: 512,
            pooling: Pooling::Mean,
        },
        dim,
    )
}

const fn unprefixed(
    pooling: Pooling,
    max_length: usize,
    dim: usize,
) -> (ModelSpec<'static>, usize) {
    (
        ModelSpec {
            passage_prefix: "",
            query_prefix: "",
            max_length,
            pooling,
        },
        dim,
    )
}

const fn embedder(
    repository: &'static str,
    revision: &'static str,
    files: [Pinned; 3],
    (spec, dim): (ModelSpec<'static>, usize),
) -> EmbedderRecord {
    EmbedderRecord {
        entry: Entry {
            repository,
            revision,
            files,
        },
        spec,
        dim,
    }
}

const BGE_BASE_EN_V1_5: EmbedderRecord = embedder(
    "BAAI/bge-base-en-v1.5",
    "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    files(
        (
            777,
            "bc00af31a4a31b74040d73370aa83b62da34c90b75eb77bfa7db039d90abd591",
        ),
        BERT_UNCASED_TOKENIZER,
        (
            437_955_512,
            "c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7",
        ),
    ),
    search(768),
);

const BGE_LARGE_EN_V1_5: EmbedderRecord = embedder(
    "BAAI/bge-large-en-v1.5",
    "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    files(
        (
            779,
            "446712fac367857b4b1302762fe1cd7bfa8b3c4b77b4dc5d77c4025407660896",
        ),
        BERT_UNCASED_TOKENIZER,
        (
            1_340_616_616,
            "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7",
        ),
    ),
    search(1024),
);

// Main holds only pytorch_model.bin; this commit is its safetensors conversion (pull request 130).
const BGE_M3: EmbedderRecord = embedder(
    "BAAI/bge-m3",
    "9a0624b896d81da7492a910ffa53731274b6cf3d",
    files(
        (
            687,
            "26159e7ad065073448460117eb24b7a4572f6f4e78eadff65dc0a11c052449fa",
        ),
        (
            17_098_108,
            "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08",
        ),
        (
            2_271_064_456,
            "993b2248881724788dcab8c644a91dfd63584b6e5604ff2037cb5541e1e38e7e",
        ),
    ),
    unprefixed(Pooling::Cls, 512, 1024),
);

const BGE_SMALL_EN_V1_5: EmbedderRecord = embedder(
    "BAAI/bge-small-en-v1.5",
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    files(
        (
            743,
            "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750",
        ),
        BERT_UNCASED_TOKENIZER,
        (
            133_466_304,
            "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
        ),
    ),
    search(384),
);

// Its weights are BF16, which widens to FP32 exactly.
const GRANITE_EMBEDDING_ENGLISH_R2: EmbedderRecord = embedder(
    "ibm-granite/granite-embedding-english-r2",
    "47ea694b257b703fee9253d75c2b1f2985180498",
    files(
        (
            1_307,
            "bc4ad4e8cd50c0069292863117823f665da70bbbdea40fd4ea77d80609ade241",
        ),
        MODERNBERT_TOKENIZER,
        (
            298_041_696,
            "23bd9625b32680f180703408ed9d8d7b3a9c1efe4c950111e3ae9d9695b21b29",
        ),
    ),
    unprefixed(Pooling::Cls, 8192, 768),
);

const E5_LARGE: EmbedderRecord = embedder(
    "intfloat/e5-large",
    "4dc6d853a804b9c8886ede6dda8a073b7dc08a81",
    files(
        (
            611,
            "d960925459889922fd6b3078e05d89c5b0fadd0018d26421699066cdd98bfb3a",
        ),
        (
            466_081,
            "5fd1c882abbd30517dced455a2c9768945ec726b96727927e4959348d9de550b",
        ),
        (
            1_340_616_616,
            "93e04bd3ec4911982905395e3c48ee739ddb8aa88380b8ce27f1fd20bfa5fa8e",
        ),
    ),
    e5(1024),
);

const E5_LARGE_V2: EmbedderRecord = embedder(
    "intfloat/e5-large-v2",
    "f169b11e22de13617baa190a028a32f3493550b6",
    files(
        (
            616,
            "2394ce5c9ffac96c88496de5d7a92f5d3fec1015b9d25f9f85545f13dfd7a269",
        ),
        BERT_UNCASED_TOKENIZER,
        (
            1_340_616_616,
            "d741c1a688a6169af0ecb5a047c44645cd992c31e1bf431269f98bba9ae2911a",
        ),
    ),
    e5(1024),
);

const E5_SMALL_V2: EmbedderRecord = embedder(
    "intfloat/e5-small-v2",
    "ffb93f3bd4047442299a41ebb6fa998a38507c52",
    files(
        (
            615,
            "5dfb0363cd0243be179c03bcaafd1542d0fbb95e8cbcf575fff3e229342adc2f",
        ),
        BERT_UNCASED_TOKENIZER,
        (
            133_466_304,
            "45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1",
        ),
    ),
    e5(384),
);

const MULTILINGUAL_E5_BASE: EmbedderRecord = embedder(
    "intfloat/multilingual-e5-base",
    "d128750597153bb5987e10b1c3493a34e5a4502a",
    files(
        (
            694,
            "9dab198f24c8c0879e481cf7822005d5ecbceedbacb390ffafa594e28d31bac4",
        ),
        MULTILINGUAL_E5_TOKENIZER,
        (
            1_112_201_288,
            "a18a44fad1d0b46ded15928144138cff1135d5cc8233bdd90be5f18822de09a7",
        ),
    ),
    e5(768),
);

const MULTILINGUAL_E5_LARGE: EmbedderRecord = embedder(
    "intfloat/multilingual-e5-large",
    "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
    files(
        (
            690,
            "e42d98d0c211928d788467aa951000391017f647c76ca8ce2ef6dd8d0c9ce5ca",
        ),
        MULTILINGUAL_E5_TOKENIZER,
        (
            2_239_611_368,
            "020afdebf2762b29fcaf286629a96c3b3b65af241f6a08226b1cfee60a21def6",
        ),
    ),
    e5(1024),
);

const MULTILINGUAL_E5_SMALL: EmbedderRecord = embedder(
    "intfloat/multilingual-e5-small",
    "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    files(
        (
            655,
            "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959",
        ),
        (
            17_082_730,
            "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
        ),
        (
            470_641_600,
            "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477",
        ),
    ),
    e5(384),
);

const MXBAI_EMBED_LARGE_V1: EmbedderRecord = embedder(
    "mixedbread-ai/mxbai-embed-large-v1",
    "b33106f585b9ce46904ad7443a3b52b7a63e231c",
    files(
        (
            677,
            "ce13f118fc183c005236901d2662ced752de632ebb161e6af800aadda2d176f7",
        ),
        BERT_UNCASED_TOKENIZER,
        (
            670_328_392,
            "36bfa45da00eb762ef9feebe4cff315ec779efadd08da11846bef5ba5b59b8f8",
        ),
    ),
    search(1024),
);

const MODERNBERT_EMBED_BASE: EmbedderRecord = embedder(
    "nomic-ai/modernbert-embed-base",
    "d556a88e332558790b210f7bdbe87da2fa94a8d8",
    files(
        (
            1_261,
            "dea4708ed1a6df7306b483037750464a96e77ccb0c7e88f4f61a0116944a71dd",
        ),
        MODERNBERT_TOKENIZER,
        (
            596_070_136,
            "69f22ccd1ee0971b6173678c4fcadd29a49a3f5b6b8a6927002418abe4bd5b9c",
        ),
    ),
    (
        ModelSpec {
            passage_prefix: "search_document: ",
            query_prefix: "search_query: ",
            max_length: 8192,
            pooling: Pooling::Mean,
        },
        768,
    ),
);

const ALL_MINILM_L12_V2: EmbedderRecord = embedder(
    "sentence-transformers/all-MiniLM-L12-v2",
    "a50ef00143b4d5391434df20ae11632588ac25be",
    files(
        (
            615,
            "bc451f333af67312ba0de5018ef1c9ba663cb18549443e568f0bd35262dc1c48",
        ),
        ALL_MINILM_TOKENIZER,
        (
            133_466_304,
            "d2d541e5f101695ae495eacd867a8d025ecfe8f9674fb23aa6cf93cdb60a5542",
        ),
    ),
    unprefixed(Pooling::Mean, 128, 384),
);

const ALL_MINILM_L6_V2: EmbedderRecord = embedder(
    "sentence-transformers/all-MiniLM-L6-v2",
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    files(
        (
            612,
            "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
        ),
        ALL_MINILM_TOKENIZER,
        (
            90_868_376,
            "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
        ),
    ),
    unprefixed(Pooling::Mean, 256, 384),
);

const ALL_MPNET_BASE_V2: EmbedderRecord = embedder(
    "sentence-transformers/all-mpnet-base-v2",
    "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    files(
        (
            571,
            "d46a3e04ded82bba22528424480697d394eeda6a27484e08c5bb2bdf5906cfa0",
        ),
        (
            466_021,
            "b8be2c30ba5dd723a6d5ee26d013da103d5408d92ddcb23747622f9e48f1d842",
        ),
        (
            437_971_872,
            "78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658",
        ),
    ),
    unprefixed(Pooling::Mean, 384, 768),
);

const PARAPHRASE_MULTILINGUAL_MINILM_L12_V2: EmbedderRecord = embedder(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
    files(
        (
            645,
            "6300193cb75e01cf80c96decef7187dfb33094d97cc1490b7ead6ff134476e4e",
        ),
        PARAPHRASE_MULTILINGUAL_TOKENIZER,
        (
            470_641_600,
            "eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b",
        ),
    ),
    unprefixed(Pooling::Mean, 128, 384),
);

const PARAPHRASE_MULTILINGUAL_MPNET_BASE_V2: EmbedderRecord = embedder(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "4328cf26390c98c5e3c738b4460a05b95f4911f5",
    files(
        (
            723,
            "8b94b5c10efd3c3ca667e5ccd501314eb670cefbf7a465f0a3fc0928a1a0ab87",
        ),
        PARAPHRASE_MULTILINGUAL_TOKENIZER,
        (
            1_112_201_288,
            "b5722100700c48b74c9c199c5f39ff9493c94d1497f719e791d4bc3861d7714a",
        ),
    ),
    unprefixed(Pooling::Mean, 128, 768),
);

const SNOWFLAKE_ARCTIC_EMBED_L: EmbedderRecord = embedder(
    "Snowflake/snowflake-arctic-embed-l",
    "d8fb21ca8d905d2832ee8b96c894d3298964346b",
    files(
        (
            704,
            "e2e01fe381e2a939eaed23dd37bdedeedfad37b034fd83c42916923f74b25a79",
        ),
        ARCTIC_TOKENIZER,
        (
            1_336_413_848,
            "46e36bf399c152c51818e1d0423bfe628d79fc9536dbf9d4fcc93078827d6c99",
        ),
    ),
    search(1024),
);

const SNOWFLAKE_ARCTIC_EMBED_M_V1_5: EmbedderRecord = embedder(
    "Snowflake/snowflake-arctic-embed-m-v1.5",
    "e58a8f756156a1293d763f17e3aae643474e9b8a",
    files(
        (
            772,
            "d2dddc06af0aeeb7fd3289a3dee1ac76dce0c3df2467cadd26f5f731f8d7919b",
        ),
        ARCTIC_TOKENIZER,
        (
            435_588_776,
            "463d27a3e88a748997316367c419bbdab4d809300a5f24b9b19ba9b2dc08ffdd",
        ),
    ),
    search(768),
);

const GTE_RERANKER_MODERNBERT_BASE: RerankerRecord = RerankerRecord {
    entry: Entry {
        repository: "Alibaba-NLP/gte-reranker-modernbert-base",
        revision: "f7481e6055501a30fb19d090657df9ec1f79ab2c",
        files: files(
            (
                1_333,
                "c9316ff715158502dad782f35454eee18de984160618dc30afe7508feb46b7ce",
            ),
            (
                3_583_499,
                "2aea6ff4701d063e7e029b6be695a1659f2caaa2ae4fb0e8b18285818271becd",
            ),
            (
                598_436_708,
                "13c533d902d6a48fa97375c6856c3e60854f9228ea760e3fc18ebde057e0cf31",
            ),
        ),
    },
    max_length: 8192,
};

const BGE_RERANKER_V2_M3: RerankerRecord = RerankerRecord {
    entry: Entry {
        repository: "BAAI/bge-reranker-v2-m3",
        revision: "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        files: files(
            (
                795,
                "13dcd6c31d9fec9d1d8e158702072f62d7fa7d312a64b9fe057bec9a08cfe41a",
            ),
            (
                17_098_273,
                "69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15",
            ),
            (
                2_271_071_852,
                "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286",
            ),
        ),
    },
    max_length: 512,
};

const MS_MARCO_MINILM_L12_V2: RerankerRecord = RerankerRecord {
    entry: Entry {
        repository: "cross-encoder/ms-marco-MiniLM-L12-v2",
        revision: "7b0235231ca2674cb8ca8f022859a6eba2b1c968",
        files: files(
            (
                791,
                "69a745055b0307584d2903ea5cf4e1899254e511834f358be80d049918b144d8",
            ),
            BERT_UNCASED_TOKENIZER,
            (
                133_469_020,
                "1ed84b90cdf3518f76ec9bb93a16f97887eea7c4e7ee5dfb03cc297e394fbbc3",
            ),
        ),
    },
    max_length: 512,
};

const MS_MARCO_MINILM_L6_V2: RerankerRecord = RerankerRecord {
    entry: Entry {
        repository: "cross-encoder/ms-marco-MiniLM-L6-v2",
        revision: "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        files: files(
            (
                794,
                "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc",
            ),
            BERT_UNCASED_TOKENIZER,
            (
                90_870_598,
                "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae",
            ),
        ),
    },
    max_length: 512,
};

const GRANITE_EMBEDDING_RERANKER_ENGLISH_R2: RerankerRecord = RerankerRecord {
    entry: Entry {
        repository: "ibm-granite/granite-embedding-reranker-english-r2",
        revision: "d09d3d6971b689bf9c23839e45a470874d46e13a",
        files: files(
            (
                1_527,
                "bcefb1b941a5860a41cfe83a527cf50172cc59cd2b6275b67cbd228b02a9e12f",
            ),
            (
                3_583_486,
                "39bc3f741c62ba10b6f412f2bfaf2ee0e3241bd3801c3a056bb4e0dbdabcf416",
            ),
            (
                598_436_708,
                "e72c8f7087dbfbdbfb9d75077f7b7b05b01b459730f735b468cbde71d66090e6",
            ),
        ),
    },
    // The length its tokenizer truncates to, which sentence-transformers follows.
    max_length: 1124,
};

/// An embedding model that loads by name: downloaded on first use, checked
/// against pinned hashes.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbeddingModel {
    /// `BAAI/bge-base-en-v1.5`: 768 values, CLS pooling, 512 tokens, a
    /// retrieval instruction before queries.
    BgeBaseEnV1_5,
    /// `BAAI/bge-large-en-v1.5`: 1,024 values, CLS pooling, 512 tokens, a
    /// retrieval instruction before queries.
    BgeLargeEnV1_5,
    /// `BAAI/bge-m3`: 1,024 values, CLS pooling, 512 tokens, no prefixes.
    BgeM3,
    /// `BAAI/bge-small-en-v1.5`: 384 values, CLS pooling, 512 tokens, a
    /// retrieval instruction before queries.
    BgeSmallEnV1_5,
    /// `ibm-granite/granite-embedding-english-r2`: ModernBERT, 768 values, CLS pooling,
    /// 8,192 tokens, no prefixes.
    GraniteEmbeddingEnglishR2,
    /// `intfloat/e5-large`: 1,024 values, 512 tokens, `passage: ` and `query: ` prefixes.
    E5Large,
    /// `intfloat/e5-large-v2`: 1,024 values, 512 tokens, `passage: ` and `query: ` prefixes.
    E5LargeV2,
    /// `intfloat/e5-small-v2`: 384 values, 512 tokens, `passage: ` and `query: ` prefixes.
    E5SmallV2,
    /// `intfloat/multilingual-e5-base`: 768 values, 512 tokens, `passage: `
    /// and `query: ` prefixes.
    MultilingualE5Base,
    /// `intfloat/multilingual-e5-large`: 1,024 values, 512 tokens, `passage: `
    /// and `query: ` prefixes.
    MultilingualE5Large,
    /// `intfloat/multilingual-e5-small`: 384 values, 512 tokens, `passage: `
    /// and `query: ` prefixes.
    MultilingualE5Small,
    /// `mixedbread-ai/mxbai-embed-large-v1`: 1,024 values, CLS pooling, 512
    /// tokens, a retrieval instruction before queries.
    MxbaiEmbedLargeV1,
    /// `nomic-ai/modernbert-embed-base`: ModernBERT, 768 values, 8,192 tokens,
    /// `search_document: ` and `search_query: ` prefixes.
    ModernBertEmbedBase,
    /// `sentence-transformers/all-MiniLM-L12-v2`: 384 values, 128 tokens, no prefixes.
    AllMiniLmL12V2,
    /// `sentence-transformers/all-MiniLM-L6-v2`: 384 values, 256 tokens, no prefixes.
    AllMiniLmL6V2,
    /// `sentence-transformers/all-mpnet-base-v2`: 768 values, 384 tokens, no prefixes.
    AllMpnetBaseV2,
    /// `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: 384
    /// values, 128 tokens, no prefixes.
    ParaphraseMultilingualMiniLmL12V2,
    /// `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`: 768
    /// values, 128 tokens, no prefixes.
    ParaphraseMultilingualMpnetBaseV2,
    /// `Snowflake/snowflake-arctic-embed-l`: 1,024 values, CLS pooling, 512
    /// tokens, a retrieval instruction before queries.
    SnowflakeArcticEmbedL,
    /// `Snowflake/snowflake-arctic-embed-m-v1.5`: 768 values, CLS pooling, 512
    /// tokens, a retrieval instruction before queries.
    SnowflakeArcticEmbedMV1_5,
}

impl EmbeddingModel {
    /// Every embedding model that loads by name, in repository-name order.
    pub const ALL: &'static [Self] = &[
        Self::BgeBaseEnV1_5,
        Self::BgeLargeEnV1_5,
        Self::BgeM3,
        Self::BgeSmallEnV1_5,
        Self::GraniteEmbeddingEnglishR2,
        Self::E5Large,
        Self::E5LargeV2,
        Self::E5SmallV2,
        Self::MultilingualE5Base,
        Self::MultilingualE5Large,
        Self::MultilingualE5Small,
        Self::MxbaiEmbedLargeV1,
        Self::ModernBertEmbedBase,
        Self::AllMiniLmL12V2,
        Self::AllMiniLmL6V2,
        Self::AllMpnetBaseV2,
        Self::ParaphraseMultilingualMiniLmL12V2,
        Self::ParaphraseMultilingualMpnetBaseV2,
        Self::SnowflakeArcticEmbedL,
        Self::SnowflakeArcticEmbedMV1_5,
    ];

    /// The Hugging Face repository, such as `intfloat/e5-large`.
    pub fn repository(self) -> &'static str {
        self.entry().repository
    }

    /// The pinned commit of [`Self::repository`].
    pub fn revision(self) -> &'static str {
        self.entry().revision
    }

    /// The prefixes, truncation and pooling the model expects.
    pub fn spec(self) -> ModelSpec<'static> {
        self.record().spec
    }

    /// Values per vector.
    pub fn dim(self) -> usize {
        self.record().dim
    }

    /// Bytes of the pinned files, which the first use downloads.
    pub fn download_size(self) -> u64 {
        self.entry().download_size()
    }

    /// Downloads the model into `hub` unless it is there, checks every file
    /// against its pinned SHA-256 and returns the directory holding them.
    pub fn download(self, hub: &Hub) -> Result<PathBuf, Error> {
        hub.fetch(self.entry())
    }

    pub(crate) fn entry(self) -> &'static Entry {
        &self.record().entry
    }

    fn record(self) -> &'static EmbedderRecord {
        match self {
            Self::BgeBaseEnV1_5 => &BGE_BASE_EN_V1_5,
            Self::BgeLargeEnV1_5 => &BGE_LARGE_EN_V1_5,
            Self::BgeM3 => &BGE_M3,
            Self::BgeSmallEnV1_5 => &BGE_SMALL_EN_V1_5,
            Self::GraniteEmbeddingEnglishR2 => &GRANITE_EMBEDDING_ENGLISH_R2,
            Self::E5Large => &E5_LARGE,
            Self::E5LargeV2 => &E5_LARGE_V2,
            Self::E5SmallV2 => &E5_SMALL_V2,
            Self::MultilingualE5Base => &MULTILINGUAL_E5_BASE,
            Self::MultilingualE5Large => &MULTILINGUAL_E5_LARGE,
            Self::MultilingualE5Small => &MULTILINGUAL_E5_SMALL,
            Self::MxbaiEmbedLargeV1 => &MXBAI_EMBED_LARGE_V1,
            Self::ModernBertEmbedBase => &MODERNBERT_EMBED_BASE,
            Self::AllMiniLmL12V2 => &ALL_MINILM_L12_V2,
            Self::AllMiniLmL6V2 => &ALL_MINILM_L6_V2,
            Self::AllMpnetBaseV2 => &ALL_MPNET_BASE_V2,
            Self::ParaphraseMultilingualMiniLmL12V2 => &PARAPHRASE_MULTILINGUAL_MINILM_L12_V2,
            Self::ParaphraseMultilingualMpnetBaseV2 => &PARAPHRASE_MULTILINGUAL_MPNET_BASE_V2,
            Self::SnowflakeArcticEmbedL => &SNOWFLAKE_ARCTIC_EMBED_L,
            Self::SnowflakeArcticEmbedMV1_5 => &SNOWFLAKE_ARCTIC_EMBED_M_V1_5,
        }
    }
}

impl FromStr for EmbeddingModel {
    type Err = Error;

    /// A repository such as `intfloat/e5-large`, in any letter case.
    fn from_str(name: &str) -> Result<Self, Error> {
        parse("embedding model", name, Self::ALL, Self::repository)
    }
}

/// A cross-encoder that loads by name, as [`EmbeddingModel`] does.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RerankerModel {
    /// `Alibaba-NLP/gte-reranker-modernbert-base`: pairs truncate to 8,192 tokens.
    GteRerankerModernBertBase,
    /// `BAAI/bge-reranker-v2-m3`: pairs truncate to 512 tokens.
    BgeRerankerV2M3,
    /// `cross-encoder/ms-marco-MiniLM-L12-v2`: pairs truncate to 512 tokens.
    MsMarcoMiniLmL12V2,
    /// `cross-encoder/ms-marco-MiniLM-L6-v2`: pairs truncate to 512 tokens.
    MsMarcoMiniLmL6V2,
    /// `ibm-granite/granite-embedding-reranker-english-r2`: pairs truncate to 1,124 tokens.
    GraniteEmbeddingRerankerEnglishR2,
}

impl RerankerModel {
    /// Every reranker that loads by name, in repository-name order.
    pub const ALL: &'static [Self] = &[
        Self::GteRerankerModernBertBase,
        Self::BgeRerankerV2M3,
        Self::MsMarcoMiniLmL12V2,
        Self::MsMarcoMiniLmL6V2,
        Self::GraniteEmbeddingRerankerEnglishR2,
    ];

    /// The Hugging Face repository, such as `cross-encoder/ms-marco-MiniLM-L6-v2`.
    pub fn repository(self) -> &'static str {
        self.entry().repository
    }

    /// The pinned commit of [`Self::repository`].
    pub fn revision(self) -> &'static str {
        self.entry().revision
    }

    /// The tokens a query and passage pair truncates to, special tokens included.
    pub fn max_length(self) -> usize {
        self.record().max_length
    }

    /// Bytes of the pinned files, as for [`EmbeddingModel::download_size`].
    pub fn download_size(self) -> u64 {
        self.entry().download_size()
    }

    /// Downloads the model into `hub`, as [`EmbeddingModel::download`] does.
    pub fn download(self, hub: &Hub) -> Result<PathBuf, Error> {
        hub.fetch(self.entry())
    }

    pub(crate) fn entry(self) -> &'static Entry {
        &self.record().entry
    }

    fn record(self) -> &'static RerankerRecord {
        match self {
            Self::GteRerankerModernBertBase => &GTE_RERANKER_MODERNBERT_BASE,
            Self::BgeRerankerV2M3 => &BGE_RERANKER_V2_M3,
            Self::MsMarcoMiniLmL12V2 => &MS_MARCO_MINILM_L12_V2,
            Self::MsMarcoMiniLmL6V2 => &MS_MARCO_MINILM_L6_V2,
            Self::GraniteEmbeddingRerankerEnglishR2 => &GRANITE_EMBEDDING_RERANKER_ENGLISH_R2,
        }
    }
}

impl FromStr for RerankerModel {
    type Err = Error;

    /// A repository such as `cross-encoder/ms-marco-MiniLM-L6-v2`, in any
    /// letter case.
    fn from_str(name: &str) -> Result<Self, Error> {
        parse("reranker", name, Self::ALL, Self::repository)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries() -> impl Iterator<Item = &'static Entry> {
        EmbeddingModel::ALL
            .iter()
            .map(|model| model.entry())
            .chain(RerankerModel::ALL.iter().map(|model| model.entry()))
    }

    fn lowercase_hex(text: &str, len: usize) -> bool {
        text.len() == len && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    }

    #[test]
    fn every_entry_pins_a_commit_and_the_files_the_loader_reads() {
        for entry in entries() {
            let (owner, name) = entry.repository.split_once('/').unwrap_or_default();
            assert!(!owner.is_empty() && !name.is_empty(), "{entry:?}");
            assert!(lowercase_hex(entry.revision, 40), "{entry:?}");
            let names = entry.files.each_ref().map(|file| file.name);
            assert_eq!(
                names,
                ["config.json", "tokenizer.json", "model.safetensors"]
            );
            for file in &entry.files {
                assert!(file.size > 0 && lowercase_hex(file.sha256, 64), "{file:?}");
            }
        }
    }

    #[test]
    fn names_parse_in_any_letter_case() {
        for &model in EmbeddingModel::ALL {
            assert_eq!(model.repository().parse::<EmbeddingModel>().unwrap(), model);
            let shouted = model.repository().to_uppercase();
            assert_eq!(shouted.parse::<EmbeddingModel>().unwrap(), model);
        }
        for &model in RerankerModel::ALL {
            assert_eq!(model.repository().parse::<RerankerModel>().unwrap(), model);
        }
        assert_eq!(
            "intfloat/E5-Large".parse::<EmbeddingModel>().unwrap(),
            EmbeddingModel::E5Large
        );
    }

    #[test]
    fn base_models_load_by_name_with_their_cards_settings() {
        for (name, model) in [
            ("BAAI/bge-base-en-v1.5", EmbeddingModel::BgeBaseEnV1_5),
            (
                "Snowflake/snowflake-arctic-embed-m-v1.5",
                EmbeddingModel::SnowflakeArcticEmbedMV1_5,
            ),
        ] {
            assert_eq!(name.parse::<EmbeddingModel>().unwrap(), model);
            let spec = model.spec();
            assert_eq!(
                (model.dim(), spec.pooling, spec.max_length),
                (768, Pooling::Cls, 512)
            );
            assert_eq!((spec.passage_prefix, spec.query_prefix), ("", SEARCH_QUERY));
        }
    }

    #[test]
    fn modernbert_models_load_by_name_with_their_cards_settings() {
        let granite = EmbeddingModel::GraniteEmbeddingEnglishR2;
        let nomic = EmbeddingModel::ModernBertEmbedBase;
        for (name, model) in [
            ("ibm-granite/granite-embedding-english-r2", granite),
            ("nomic-ai/modernbert-embed-base", nomic),
        ] {
            assert_eq!(name.parse::<EmbeddingModel>().unwrap(), model);
            assert_eq!((model.dim(), model.spec().max_length), (768, 8192));
        }
        assert_eq!(
            (
                granite.spec().pooling,
                granite.spec().passage_prefix,
                granite.spec().query_prefix
            ),
            (Pooling::Cls, "", "")
        );
        assert_eq!(
            (
                nomic.spec().pooling,
                nomic.spec().passage_prefix,
                nomic.spec().query_prefix
            ),
            (Pooling::Mean, "search_document: ", "search_query: ")
        );
        assert_eq!(
            granite.entry().files[1].sha256,
            nomic.entry().files[1].sha256
        );
    }

    #[test]
    fn modernbert_rerankers_load_by_name_with_their_pairs_lengths() {
        for (name, model, max_length) in [
            (
                "Alibaba-NLP/gte-reranker-modernbert-base",
                RerankerModel::GteRerankerModernBertBase,
                8192,
            ),
            (
                "ibm-granite/granite-embedding-reranker-english-r2",
                RerankerModel::GraniteEmbeddingRerankerEnglishR2,
                1124,
            ),
        ] {
            assert_eq!(name.parse::<RerankerModel>().unwrap(), model);
            assert_eq!(model.max_length(), max_length);
        }
    }

    #[test]
    fn unknown_names_list_the_models_that_load() {
        let Err(Error::Unsupported(reason)) = "BAAI/bge-base-en".parse::<EmbeddingModel>() else {
            panic!("an unregistered model must not parse");
        };
        assert!(reason.contains("BAAI/bge-base-en"), "{reason}");
        for &model in EmbeddingModel::ALL {
            assert!(reason.contains(model.repository()), "{reason}");
        }
        assert!(
            "cross-encoder/ms-marco-MiniLM-L6-v2"
                .parse::<EmbeddingModel>()
                .is_err()
        );
        assert!("intfloat/e5-large".parse::<RerankerModel>().is_err());
    }

    #[test]
    fn a_mismatch_names_the_file_both_hashes_and_the_remedy() {
        let reason = mismatch(Path::new("cache/config.json"), "aa", "bb");
        assert!(reason.contains("config.json") && reason.contains("aa") && reason.contains("bb"));
        assert!(reason.contains("delete"), "{reason}");
    }

    #[test]
    fn download_sizes_add_up_the_pinned_files() {
        // config.json, tokenizer.json and model.safetensors as the Hub lists them.
        assert_eq!(
            EmbeddingModel::AllMiniLmL6V2.download_size(),
            612 + 466_247 + 90_868_376
        );
        assert_eq!(
            EmbeddingModel::E5Large.download_size(),
            611 + 466_081 + 1_340_616_616
        );
        assert_eq!(
            RerankerModel::MsMarcoMiniLmL6V2.download_size(),
            794 + 711_396 + 90_870_598
        );
    }

    #[test]
    fn repositories_are_distinct() {
        let mut repositories: Vec<&str> = entries().map(|entry| entry.repository).collect();
        repositories.sort_unstable();
        repositories.dedup();
        assert_eq!(repositories.len(), entries().count());
    }

    #[test]
    fn models_are_listed_in_repository_name_order() {
        let ordered = |names: &[&str]| {
            names
                .windows(2)
                .all(|pair| pair[0].to_lowercase() < pair[1].to_lowercase())
        };
        let embedders: Vec<&str> = EmbeddingModel::ALL.iter().map(|m| m.repository()).collect();
        let rerankers: Vec<&str> = RerankerModel::ALL.iter().map(|m| m.repository()).collect();
        assert!(ordered(&embedders), "{embedders:?}");
        assert!(ordered(&rerankers), "{rerankers:?}");
    }

    #[test]
    fn each_embedder_keeps_its_own_text_settings_and_width() {
        let minilm = EmbeddingModel::AllMiniLmL6V2;
        let e5 = EmbeddingModel::E5Large;
        let bge = EmbeddingModel::BgeSmallEnV1_5;
        assert_eq!(
            minilm.repository(),
            "sentence-transformers/all-MiniLM-L6-v2"
        );
        assert_eq!((minilm.dim(), minilm.spec().max_length), (384, 256));
        assert_eq!(
            (minilm.spec().passage_prefix, minilm.spec().query_prefix),
            ("", "")
        );
        assert_eq!(e5.repository(), "intfloat/e5-large");
        assert_eq!((e5.dim(), e5.spec().max_length), (1024, 512));
        assert_eq!(
            (e5.spec().passage_prefix, e5.spec().query_prefix),
            ("passage: ", "query: ")
        );
        assert_eq!(
            (minilm.spec().pooling, e5.spec().pooling),
            (Pooling::Mean, Pooling::Mean)
        );
        assert_eq!(bge.repository(), "BAAI/bge-small-en-v1.5");
        assert_eq!((bge.dim(), bge.spec().pooling), (384, Pooling::Cls));
        assert_eq!(
            (bge.spec().passage_prefix, bge.spec().query_prefix),
            ("", SEARCH_QUERY)
        );
    }
}
