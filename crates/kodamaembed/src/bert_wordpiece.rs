//! A guarded shortcut to BERT's token IDs; it snapshots the tokenizer, so configure that first.

use rustc_hash::FxHashMap;
use serde_json::{Value, json};
use tokenizers::{Tokenizer, TruncationDirection, TruncationStrategy};

const SPECIAL: [(&str, u32); 5] = [
    ("[PAD]", 0),
    ("[UNK]", 100),
    ("[CLS]", 101),
    ("[SEP]", 102),
    ("[MASK]", 103),
];

pub(crate) struct AsciiBert {
    first: FxHashMap<Vec<u8>, u32>,
    continuing: FxHashMap<Vec<u8>, u32>,
    maximum: usize,
}

impl AsciiBert {
    /// None, also on any parse failure, selects the normal tokenizer path.
    pub(crate) fn from_tokenizer(tokenizer: &Tokenizer) -> Option<Self> {
        let truncation = tokenizer.get_truncation()?;
        if tokenizer.get_padding().is_some()
            || tokenizer.get_encode_special_tokens()
            || truncation.direction != TruncationDirection::Right
            || truncation.strategy != TruncationStrategy::LongestFirst
            || truncation.stride != 0
            || truncation.max_length < 2
        {
            return None;
        }
        // The live configuration, not checkpoint JSON, which can be stale.
        let configuration = serde_json::to_value(tokenizer).ok()?;
        Self::from_configuration(&configuration, truncation.max_length)
    }

    fn from_configuration(v: &Value, maximum: usize) -> Option<Self> {
        if maximum < 2
            || v["normalizer"]
                != json!({
                    "type":"BertNormalizer", "clean_text":true,
                    "handle_chinese_chars":true, "strip_accents":null, "lowercase":true
                })
            || v["pre_tokenizer"] != json!({"type":"BertPreTokenizer"})
            || v["model"]["type"] != "WordPiece"
            || v["model"]["unk_token"] != "[UNK]"
            || v["model"]["continuing_subword_prefix"] != "##"
            || v["model"]["max_input_chars_per_word"] != 100
        {
            return None;
        }
        let added: Vec<_> = SPECIAL
            .iter()
            .map(|(content, id)| {
                json!({
                    "id":id, "content":content, "special":true, "single_word":false,
                    "lstrip":false, "rstrip":false, "normalized":false
                })
            })
            .collect();
        if v["added_tokens"] != json!(added)
            || v["post_processor"]["type"] != "TemplateProcessing"
            || v["post_processor"]["single"]
                != json!([
                    {"SpecialToken":{"id":"[CLS]","type_id":0}},
                    {"Sequence":{"id":"A","type_id":0}},
                    {"SpecialToken":{"id":"[SEP]","type_id":0}}
                ])
            || v["post_processor"]["special_tokens"]
                != json!({
                    "[CLS]":{"id":"[CLS]","ids":[101],"tokens":["[CLS]"]},
                    "[SEP]":{"id":"[SEP]","ids":[102],"tokens":["[SEP]"]}
                })
        {
            return None;
        }
        let mut first = FxHashMap::default();
        let mut continuing = FxHashMap::default();
        for (word, value) in v["model"]["vocab"].as_object()? {
            let id = u32::try_from(value.as_u64()?).ok()?;
            if let Some(suffix) = word.strip_prefix("##") {
                continuing.insert(suffix.as_bytes().to_vec(), id);
            } else {
                first.insert(word.as_bytes().to_vec(), id);
            }
        }
        for (word, id) in SPECIAL {
            if first.get(word.as_bytes()) != Some(&id) {
                return None;
            }
        }
        Some(Self {
            first,
            continuing,
            maximum,
        })
    }

    fn wordpiece(&self, word: &[u8], ids: &mut Vec<u32>, limit: usize) {
        let begin = ids.len();
        if word.len() > 100 {
            ids.push(100);
            return;
        }
        let mut pieces = [0u32; 100];
        let (mut count, mut start) = (0, 0);
        while start < word.len() {
            let vocab = if start == 0 {
                &self.first
            } else {
                &self.continuing
            };
            let found = (start + 1..=word.len())
                .rev()
                .find_map(|end| vocab.get(&word[start..end]).map(|&id| (end, id)));
            match found {
                Some((end, id)) => {
                    pieces[count] = id;
                    count += 1;
                    start = end;
                }
                None => {
                    ids.push(100);
                    return;
                }
            }
        }
        ids.extend_from_slice(&pieces[..count.min(limit - begin)]);
    }

    /// None when the ASCII scanner cannot finish the kept prefix; later text need not be ASCII.
    pub(crate) fn encode(&self, text: &str) -> Option<Vec<u32>> {
        self.encode_with_limit(text, self.maximum)
    }

    pub(crate) fn maximum(&self) -> usize {
        self.maximum
    }

    /// The prefix keeps its CLS and SEP: the caller strips them and wraps the joined fragments.
    pub(crate) fn encode_with_limit(&self, text: &str, maximum: usize) -> Option<Vec<u32>> {
        if maximum < 2 || maximum > self.maximum {
            return None;
        }
        let content_end = maximum - 1;
        let mut ids = Vec::with_capacity(maximum.min(256));
        ids.push(101);
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() && ids.len() < content_end {
            let ch = bytes[i];
            if !ch.is_ascii() {
                return None;
            }
            if matches!(ch, b' ' | b'\t' | b'\n' | b'\r') || ch.is_ascii_control() {
                i += 1;
                continue;
            }
            if ch == b'['
                && let Some(&(literal, id)) = SPECIAL
                    .iter()
                    .find(|(literal, _)| bytes[i..].starts_with(literal.as_bytes()))
            {
                ids.push(id);
                i += literal.len();
                continue;
            }
            if ch.is_ascii_punctuation() {
                self.wordpiece(&[ch], &mut ids, content_end);
                i += 1;
                continue;
            }
            let mut word = [0u8; 101];
            let mut length = 0;
            while i < bytes.len() {
                let ch = bytes[i];
                if !ch.is_ascii() {
                    return None;
                }
                if matches!(ch, b' ' | b'\t' | b'\n' | b'\r') || ch.is_ascii_punctuation() {
                    break;
                }
                i += 1;
                if ch.is_ascii_control() {
                    continue;
                }
                if length < 101 {
                    word[length] = ch.to_ascii_lowercase();
                    length += 1;
                }
            }
            if length > 0 {
                self.wordpiece(&word[..length], &mut ids, content_end);
            }
        }
        ids.push(102);
        Some(ids)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use tokenizers::{PaddingParams, TruncationParams};

    fn configuration() -> Value {
        json!({
            "version":"1.0", "truncation":null, "padding":null,
            "added_tokens": SPECIAL.iter().map(|(content, id)| json!({
                "id":id,"content":content,"special":true,"single_word":false,
                "lstrip":false,"rstrip":false,"normalized":false
            })).collect::<Vec<_>>(),
            "normalizer":{"type":"BertNormalizer","clean_text":true,
                "handle_chinese_chars":true,"strip_accents":null,"lowercase":true},
            "pre_tokenizer":{"type":"BertPreTokenizer"}, "decoder":null,
            "post_processor":{
                "type":"TemplateProcessing",
                "single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},
                    {"Sequence":{"id":"A","type_id":0}},
                    {"SpecialToken":{"id":"[SEP]","type_id":0}}],
                "pair":[{"SpecialToken":{"id":"[CLS]","type_id":0}},
                    {"Sequence":{"id":"A","type_id":0}},
                    {"SpecialToken":{"id":"[SEP]","type_id":0}},
                    {"Sequence":{"id":"B","type_id":1}},
                    {"SpecialToken":{"id":"[SEP]","type_id":1}}],
                "special_tokens":{
                    "[CLS]":{"id":"[CLS]","ids":[101],"tokens":["[CLS]"]},
                    "[SEP]":{"id":"[SEP]","ids":[102],"tokens":["[SEP]"]}}
            },
            "model":{"type":"WordPiece","unk_token":"[UNK]",
                "continuing_subword_prefix":"##","max_input_chars_per_word":100,
                "vocab":{"[PAD]":0,"[UNK]":100,"[CLS]":101,"[SEP]":102,
                    "[MASK]":103,"hello":2000,"world":2001,"##s":2002}}
        })
    }

    pub(crate) fn configured() -> Tokenizer {
        let mut tokenizer: Tokenizer = serde_json::from_value(configuration()).unwrap();
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 8,
                ..TruncationParams::default()
            }))
            .unwrap();
        tokenizer
    }

    #[test]
    fn current_runtime_configuration_is_required() {
        let tokenizer = configured();
        let fast = AsciiBert::from_tokenizer(&tokenizer).unwrap();
        assert_eq!(fast.maximum, 8);
        assert_eq!(
            fast.encode("HELLO worlds"),
            Some(vec![101, 2000, 2001, 2002, 102])
        );
        let mut other = tokenizer.clone();
        other.with_truncation(None).unwrap();
        assert!(AsciiBert::from_tokenizer(&other).is_none());
        let mut other = tokenizer.clone();
        other.with_padding(Some(PaddingParams::default()));
        assert!(AsciiBert::from_tokenizer(&other).is_none());
        let mut other = tokenizer.clone();
        other.set_encode_special_tokens(true);
        assert!(AsciiBert::from_tokenizer(&other).is_none());
        let mut other = tokenizer.clone();
        other.get_truncation_mut().unwrap().direction = TruncationDirection::Left;
        assert!(AsciiBert::from_tokenizer(&other).is_none());
        for strategy in [
            TruncationStrategy::OnlyFirst,
            TruncationStrategy::OnlySecond,
        ] {
            let mut other = tokenizer.clone();
            other.get_truncation_mut().unwrap().strategy = strategy;
            assert!(AsciiBert::from_tokenizer(&other).is_none());
        }
        let mut other = tokenizer.clone();
        other.get_truncation_mut().unwrap().stride = 1;
        assert!(AsciiBert::from_tokenizer(&other).is_none());
        for maximum in [0, 1] {
            let mut other = tokenizer.clone();
            other.get_truncation_mut().unwrap().max_length = maximum;
            assert!(AsciiBert::from_tokenizer(&other).is_none());
        }
    }

    #[test]
    fn unsupported_configuration_and_bad_ids_select_fallback() {
        let original = configuration();
        assert!(AsciiBert::from_configuration(&original, 8).is_some());
        assert!(AsciiBert::from_configuration(&Value::Null, 8).is_none());
        assert!(AsciiBert::from_configuration(&original, 1).is_none());
        for (pointer, value) in [
            ("/normalizer/lowercase", json!(false)),
            ("/normalizer/clean_text", json!(false)),
            ("/normalizer/handle_chinese_chars", json!(false)),
            ("/normalizer/strip_accents", json!(false)),
            ("/normalizer/type", json!("Sequence")),
            ("/pre_tokenizer/type", json!("Whitespace")),
            ("/model/type", json!("BPE")),
            ("/model/unk_token", json!("unknown")),
            ("/model/continuing_subword_prefix", json!("@@")),
            ("/model/max_input_chars_per_word", json!(101)),
            ("/post_processor/type", json!("BertProcessing")),
            ("/post_processor/single/1/Sequence/type_id", json!(1)),
            ("/post_processor/special_tokens/[CLS]/ids/0", json!(7)),
            ("/added_tokens/0/content", json!("[NEW]")),
            ("/added_tokens/0/id", json!(7)),
            ("/added_tokens/0/special", json!(false)),
            ("/added_tokens/0/single_word", json!(true)),
            ("/added_tokens/0/lstrip", json!(true)),
            ("/added_tokens/0/rstrip", json!(true)),
            ("/added_tokens/0/normalized", json!(true)),
            ("/model/vocab/[PAD]", json!(7)),
            ("/model/vocab/hello", json!(u64::from(u32::MAX) + 1)),
            ("/model/vocab/hello", json!(-1)),
            ("/model/vocab/hello", json!(1.5)),
            ("/model/vocab/hello", json!("2000")),
            ("/model/vocab/hello", Value::Null),
        ] {
            let mut changed = original.clone();
            *changed.pointer_mut(pointer).unwrap() = value;
            assert!(
                AsciiBert::from_configuration(&changed, 8).is_none(),
                "{pointer}"
            );
        }
        let mut missing = original.clone();
        missing["model"]["vocab"]
            .as_object_mut()
            .unwrap()
            .remove("[MASK]");
        assert!(AsciiBert::from_configuration(&missing, 8).is_none());
        let mut extra = original;
        extra["added_tokens"].as_array_mut().unwrap().push(json!({
            "id":2003,"content":"extra","special":false,"single_word":false,
            "lstrip":false,"rstrip":false,"normalized":true
        }));
        assert!(AsciiBert::from_configuration(&extra, 8).is_none());
    }

    #[test]
    fn complete_word_decision_precedes_quota_and_unicode_fallback() {
        let v = configuration();
        let fast = AsciiBert::from_configuration(&v, 3).unwrap();
        // An unknown continuation makes the whole word UNK before truncation.
        assert_eq!(fast.encode("helloq"), Some(vec![101, 100, 102]));
        assert_eq!(fast.encode(&"a".repeat(101)), Some(vec![101, 100, 102]));
        assert_eq!(fast.encode("hello café"), Some(vec![101, 2000, 102]));
        assert!(fast.encode("hello\u{301}").is_none());
        assert_eq!(fast.encode("[MASK]"), Some(vec![101, 103, 102]));
        assert_eq!(fast.encode("hel\0lo"), Some(vec![101, 2000, 102]));
        let expected = configured()
            .encode_fast("HELLO [MASK] worlds", true)
            .unwrap();
        let fast = AsciiBert::from_configuration(&v, 8).unwrap();
        assert_eq!(
            fast.encode("HELLO [MASK] worlds").unwrap(),
            expected.get_ids()
        );
    }
}
