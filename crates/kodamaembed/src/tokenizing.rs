//! Inference tokenization needs IDs and segment types, but no text offsets.

use crate::bert_wordpiece::AsciiBert;
use kodamaembed_cpu::Workers;
use rayon::prelude::*;
use tokenizers::{
    EncodeInput, Encoding, InputSequence, PostProcessor, Tokenizer, TruncationParams,
    models::ModelWrapper,
    normalizers::{self, NormalizerWrapper, Replace, replace::ReplacePattern},
    pre_tokenizers::{
        PreTokenizerWrapper,
        metaspace::{Metaspace, PrependScheme},
        sequence::Sequence,
        whitespace::WhitespaceSplit,
    },
    processors::PostProcessorWrapper,
    utils::truncation::truncate_encodings,
};

// Inline, the reference Encoding costs no heap allocation per input.
#[allow(clippy::large_enum_variant)]
pub(crate) enum InferenceInput {
    Original(Encoding),
    Compact { ids: Vec<u32>, types: Vec<u32> },
}

impl InferenceInput {
    pub(crate) fn get_ids(&self) -> &[u32] {
        match self {
            Self::Original(encoding) => encoding.get_ids(),
            Self::Compact { ids, .. } => ids,
        }
    }

    pub(crate) fn get_type_ids(&self) -> &[u32] {
        match self {
            Self::Original(encoding) => encoding.get_type_ids(),
            Self::Compact { types, .. } => types,
        }
    }
}

/// Owns the configured tokenizer so its checked snapshot cannot become stale.
pub(crate) struct InferenceTokenizer {
    reference: Tokenizer,
    ascii: Option<AsciiBert>,
    pair: Option<PairTokenizer>,
}

struct PairTokenizer {
    raw: Tokenizer,
    truncation: TruncationParams,
}

impl PairTokenizer {
    fn new(reference: &Tokenizer) -> Option<Self> {
        if reference.get_padding().is_some() {
            return None;
        }
        let mut truncation = reference.get_truncation()?.clone();
        let special = reference
            .get_post_processor()
            .map_or(0, |processor| processor.added_tokens(true));
        truncation.max_length = truncation.max_length.checked_sub(special)?;
        let mut raw = reference.clone();
        raw.with_truncation(None).ok()?;
        raw.with_post_processor(None::<PostProcessorWrapper>);
        Some(Self { raw, truncation })
    }

    fn encode(
        &self,
        reference: &Tokenizer,
        first: InputSequence<'_>,
        second: InputSequence<'_>,
    ) -> tokenizers::Result<Encoding> {
        let first = self.raw.encode_fast(EncodeInput::Single(first), false)?;
        let mut second = self.raw.encode_fast(EncodeInput::Single(second), false)?;
        // Without a post-processor nothing else sets the second text's segment.
        second.set_type_ids(vec![1; second.len()]);
        let (mut first, mut second) = truncate_encodings(first, Some(second), &self.truncation)?;
        // Only the first window is used: overflows would be postprocessed as a Cartesian product.
        first.set_overflowing(Vec::new());
        if let Some(second) = &mut second {
            second.set_overflowing(Vec::new());
        }
        reference.post_process(first, second, true)
    }
}

impl InferenceTokenizer {
    pub(crate) fn new(reference: Tokenizer) -> Self {
        let ascii = AsciiBert::from_tokenizer(&reference);
        let pair = PairTokenizer::new(&reference);
        Self {
            reference,
            ascii,
            pair,
        }
    }

    /// A lone fast-path input stays on the caller: workers cannot split it.
    pub(crate) fn encode_batch_in<'s, E>(
        &self,
        pool: &Workers,
        inputs: Vec<E>,
    ) -> tokenizers::Result<Vec<InferenceInput>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        if inputs.len() == 1 && !self.general() {
            self.encode_batch(inputs)
        } else {
            pool.install(|| self.encode_batch(inputs))
        }
    }

    fn general(&self) -> bool {
        self.ascii.is_none() && self.pair.is_none()
    }

    pub(crate) fn encode_batch<'s, E>(
        &self,
        inputs: Vec<E>,
    ) -> tokenizers::Result<Vec<InferenceInput>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        if self.general() {
            // Keep batch postprocessing, including padding, on the general path.
            return Ok(encode_batch(&self.reference, inputs)?
                .into_iter()
                .map(InferenceInput::Original)
                .collect());
        }
        let encode = |input: E| -> tokenizers::Result<InferenceInput> {
            match input.into() {
                EncodeInput::Single(InputSequence::Raw(text)) if self.ascii.is_some() => {
                    let fast = self.ascii.as_ref().expect("checked ASCII tokenizer");
                    let ids = match fast.encode(&text) {
                        Some(ids) => ids,
                        None => self.fragment_ids(fast, &text)?,
                    };
                    let types = vec![0; ids.len()];
                    Ok(InferenceInput::Compact { ids, types })
                }
                EncodeInput::Dual(first, second) if self.pair.is_some() => {
                    let pair = self.pair.as_ref().expect("checked pair tokenizer");
                    Ok(InferenceInput::Original(pair.encode(
                        &self.reference,
                        first,
                        second,
                    )?))
                }
                other => Ok(InferenceInput::Original(
                    if self.reference.get_post_processor().is_some() {
                        self.reference.encode_fast(other, true)?
                    } else {
                        self.reference.encode(other, true)?
                    },
                )),
            }
        };
        if inputs.len() == 1 {
            inputs.into_iter().map(encode).collect()
        } else {
            inputs.into_par_iter().map(encode).collect()
        }
    }

    fn fragment_ids(&self, fast: &AsciiBert, text: &str) -> tokenizers::Result<Vec<u32>> {
        let maximum = fast.maximum();
        let mut ids = Vec::with_capacity(maximum.min(256));
        ids.push(101);
        // Only these four are safe boundaries: BERT deletes other controls, which joins letters.
        let mut offset = 0;
        let mut fragments = text
            .split([' ', '\t', '\n', '\r'])
            .map(|fragment| {
                let start = offset;
                // Every delimiter is one byte; even empty fragments advance it.
                offset += fragment.len() + 1;
                (fragment, start, start + fragment.len())
            })
            .filter(|(fragment, _, _)| !fragment.is_empty())
            .peekable();
        while ids.len() < maximum - 1 {
            let Some((fragment, start, mut end)) = fragments.next() else {
                break;
            };
            let remaining = maximum - 1 - ids.len();
            if let Some(part) = fast.encode_with_limit(fragment, remaining + 2) {
                ids.extend_from_slice(&part[1..part.len() - 1]);
            } else {
                // Unicode fragments share one reference setup; an ASCII one takes the shortcut.
                for _ in 1..16 {
                    if fragments.peek().is_none_or(|(next, _, _)| next.is_ascii()) {
                        break;
                    }
                    end = fragments.next().expect("peeked fragment").2;
                }
                // Whole fragments keep whole-word UNK decisions; CLS and SEP come once, at the end.
                let part = self.reference.encode_fast(&text[start..end], false)?;
                ids.extend(part.get_ids().iter().copied().take(remaining));
            }
        }
        ids.push(102);
        Ok(ids)
    }
}

/// SentencePiece vocabularies under Metaspace alone split as transformers' XLM-R tokenizer does.
pub(crate) fn sentencepiece_pipeline(tokenizer: &mut Tokenizer) {
    let metaspace = Metaspace::new('▁', PrependScheme::Always, true);
    let converted = matches!(tokenizer.get_model(), ModelWrapper::Unigram(_))
        && matches!(
            tokenizer.get_pre_tokenizer(),
            Some(PreTokenizerWrapper::Metaspace(own)) if *own == metaspace
        );
    if !converted {
        return;
    }
    // The converters' Metaspace alone emits a `▁` for a space before a special token or at the end,
    // which SentencePiece and transformers never do; the split also does the closing steps' work.
    let mut steps = match tokenizer.get_normalizer() {
        Some(NormalizerWrapper::Sequence(steps)) => steps.as_ref().to_vec(),
        Some(step) => vec![step.clone()],
        None => Vec::new(),
    };
    while steps.last().is_some_and(only_whitespace) {
        steps.pop();
    }
    tokenizer.with_normalizer(match steps.len() {
        0 => None,
        1 => steps.pop(),
        _ => Some(NormalizerWrapper::Sequence(normalizers::Sequence::new(
            steps,
        ))),
    });
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
        PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
        PreTokenizerWrapper::Metaspace(metaspace),
    ]))));
}

/// A step that trims the text or collapses its runs of spaces, as the converters write them.
fn only_whitespace(step: &NormalizerWrapper) -> bool {
    match step {
        NormalizerWrapper::StripNormalizer(_) => true,
        NormalizerWrapper::Replace(replace) => {
            matches!(replace.content.as_str(), " " | "▁")
                && Replace::new(
                    ReplacePattern::Regex(" {2,}".into()),
                    replace.content.as_str(),
                )
                .is_ok_and(|runs| runs == *replace)
        }
        _ => false,
    }
}

pub(crate) fn encode_batch<'s, E>(
    tokenizer: &Tokenizer,
    inputs: Vec<E>,
) -> tokenizers::Result<Vec<Encoding>>
where
    E: Into<EncodeInput<'s>> + Send,
{
    // tokenizers 0.22.2: encode_fast zeroes segment IDs, and only a postprocessor reinstates them.
    if tokenizer.get_post_processor().is_some() {
        tokenizer.encode_batch_fast(inputs, true)
    } else {
        tokenizer.encode_batch(inputs, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{
        AddedToken, TruncationDirection, TruncationParams, TruncationStrategy,
        models::{unigram::Unigram, wordpiece::WordPiece},
        normalizers::{Lowercase, Precompiled, Strip, bert::BertNormalizer},
        pre_tokenizers::bert::BertPreTokenizer,
        processors::{bert::BertProcessing, template::TemplateProcessing},
    };

    fn tokenizer(processor: bool) -> Tokenizer {
        let words = [
            "[UNK]", "[CLS]", "[SEP]", "hello", "world", "cafe", "!", "東", "京",
        ];
        let vocab: [(String, u32); 9] = std::array::from_fn(|id| (words[id].to_owned(), id as u32));
        let model = WordPiece::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_normalizer(Some(BertNormalizer::new(true, true, None, true)));
        tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
        if processor {
            tokenizer.with_post_processor(Some(BertProcessing::new(
                ("[SEP]".into(), 2),
                ("[CLS]".into(), 1),
            )));
        }
        tokenizer
    }

    /// A charsmap whose 256-unit trie matches no byte, so it leaves text unchanged.
    fn identity_charsmap() -> Precompiled {
        let mut charsmap = 1024u32.to_le_bytes().to_vec();
        charsmap.resize(4 + 1024, 0);
        Precompiled::from(&charsmap).unwrap()
    }

    /// XLM-R as newer transformers 4 converters write it: Metaspace alone after collapsing spaces.
    fn converted_sentencepiece() -> Tokenizer {
        let specials = ["<s>", "<pad>", "</s>", "<unk>"].map(|piece| (piece.to_owned(), 0.0));
        let words = ["▁a", "▁b", "▁"].map(|piece| (piece.to_owned(), -1.0));
        let vocab = specials.into_iter().chain(words).collect();
        let mut tokenizer = Tokenizer::new(Unigram::from(vocab, Some(3), false).unwrap());
        tokenizer.with_normalizer(Some(NormalizerWrapper::Sequence(
            tokenizers::normalizers::Sequence::new(vec![
                NormalizerWrapper::Precompiled(identity_charsmap()),
                NormalizerWrapper::Replace(
                    Replace::new(ReplacePattern::Regex(" {2,}".into()), " ").unwrap(),
                ),
            ]),
        )));
        tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Always, true)));
        tokenizer.add_special_tokens(&[AddedToken::from("<pad>", true)]);
        tokenizer
    }

    fn ids(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
        tokenizer.encode(text, false).unwrap().get_ids().to_vec()
    }

    #[test]
    fn sentencepiece_vocabularies_split_as_transformers_rebuilds_them() {
        let mut tokenizer = converted_sentencepiece();
        assert_eq!(ids(&tokenizer, "a <pad> b "), [4, 6, 1, 5, 6]);
        sentencepiece_pipeline(&mut tokenizer);
        assert_eq!(ids(&tokenizer, "a <pad> b "), [4, 1, 5]);
        assert!(matches!(
            tokenizer.get_normalizer(),
            Some(NormalizerWrapper::Precompiled(_))
        ));
    }

    /// Only the whitespace steps that close the normalizer go: the split does their work.
    #[test]
    fn sentencepiece_vocabularies_keep_their_own_normalizer_steps() {
        let pieces = ["<unk>", "▁", "▁abc", "▁ABC"].map(|piece| (piece.to_owned(), -1.0));
        let mut tokenizer = Tokenizer::new(Unigram::from(pieces.to_vec(), Some(0), false).unwrap());
        let collapse = Replace::new(ReplacePattern::Regex(" {2,}".into()), "▁").unwrap();
        tokenizer.with_normalizer(Some(NormalizerWrapper::Sequence(
            tokenizers::normalizers::Sequence::new(vec![
                NormalizerWrapper::StripNormalizer(Strip::new(true, false)),
                NormalizerWrapper::Lowercase(Lowercase),
                NormalizerWrapper::StripNormalizer(Strip::new(false, true)),
                NormalizerWrapper::Replace(collapse),
            ]),
        )));
        tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Always, true)));
        sentencepiece_pipeline(&mut tokenizer);
        assert_eq!(ids(&tokenizer, " ABC  abc "), [2, 2]);
        let Some(NormalizerWrapper::Sequence(steps)) = tokenizer.get_normalizer() else {
            panic!("the normalizer keeps two steps");
        };
        assert!(matches!(
            steps.as_ref(),
            [
                NormalizerWrapper::StripNormalizer(_),
                NormalizerWrapper::Lowercase(_)
            ]
        ));
    }

    /// A split other than the converters' Metaspace alone is the checkpoint's own choice.
    #[test]
    fn sentencepiece_vocabularies_with_another_split_keep_their_tokenizer_json_pipeline() {
        let split = PreTokenizerWrapper::Sequence(Sequence::new(vec![
            PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
            PreTokenizerWrapper::Metaspace(Metaspace::new('▁', PrependScheme::Always, true)),
        ]));
        let first = PreTokenizerWrapper::Metaspace(Metaspace::new('▁', PrependScheme::First, true));
        for own in [split, first] {
            let mut tokenizer = converted_sentencepiece();
            tokenizer.with_pre_tokenizer(Some(own));
            let before = tokenizer.to_string(false).unwrap();
            sentencepiece_pipeline(&mut tokenizer);
            assert_eq!(tokenizer.to_string(false).unwrap(), before);
        }
    }

    #[test]
    fn other_vocabularies_keep_their_tokenizer_json_pipeline() {
        let mut tokenizer = tokenizer(false);
        let before = ids(&tokenizer, "Hello  CAFÉ 東京! ");
        sentencepiece_pipeline(&mut tokenizer);
        assert_eq!(ids(&tokenizer, "Hello  CAFÉ 東京! "), before);
    }

    fn equal_inputs(actual: &[Encoding], expected: &[Encoding]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.get_ids(), expected.get_ids());
            assert_eq!(actual.get_type_ids(), expected.get_type_ids());
            assert_eq!(actual.get_attention_mask(), expected.get_attention_mask());
        }
    }

    #[test]
    fn inference_inputs_match_for_unicode_pairs_special_tokens_and_truncation() {
        let long = "hello world ".repeat(40);
        for processor in [false, true] {
            for direction in [TruncationDirection::Left, TruncationDirection::Right] {
                let mut tokenizer = tokenizer(processor);
                tokenizer
                    .with_truncation(Some(TruncationParams {
                        max_length: 8,
                        direction,
                        ..Default::default()
                    }))
                    .unwrap();
                let texts = ["", "Hello CAFÉ 東京!", "[CLS] hello [SEP]", long.as_str()];
                equal_inputs(
                    &encode_batch(&tokenizer, texts.to_vec()).unwrap(),
                    &tokenizer.encode_batch(texts.to_vec(), true).unwrap(),
                );
                let pairs: Vec<_> = texts
                    .iter()
                    .enumerate()
                    .map(|(index, text)| (*text, texts[(index + 1) % texts.len()]))
                    .collect();
                equal_inputs(
                    &encode_batch(&tokenizer, pairs.clone()).unwrap(),
                    &tokenizer.encode_batch(pairs, true).unwrap(),
                );
            }
        }
    }

    #[test]
    fn pairs_without_postprocessing_preserve_second_segment_ids() {
        let tokenizer = tokenizer(false);
        let encoded = encode_batch(&tokenizer, vec![("hello world", "cafe world")]).unwrap();
        assert_eq!(encoded[0].get_ids(), &[3, 4, 5, 4]);
        assert_eq!(encoded[0].get_type_ids(), &[0, 0, 1, 1]);
    }

    #[test]
    fn inference_pairs_discard_overflow_before_cartesian_merge() {
        let mut reference = tokenizer(true);
        reference
            .with_truncation(Some(TruncationParams {
                max_length: 32,
                ..Default::default()
            }))
            .unwrap();
        let text = "hello world ".repeat(128);
        let expected = reference
            .encode_fast((text.as_str(), text.as_str()), true)
            .unwrap();
        assert!(expected.get_overflowing().len() > 100);
        let inference = InferenceTokenizer::new(reference);
        let actual = inference
            .encode_batch(vec![(text.as_str(), text.as_str())])
            .unwrap();
        equal_compact_inputs(&actual, std::slice::from_ref(&expected));
        let InferenceInput::Original(actual) = &actual[0] else {
            panic!("expected pair encoding")
        };
        assert!(
            actual.get_overflowing().is_empty(),
            "retained {} discarded pair windows",
            actual.get_overflowing().len()
        );
    }

    #[test]
    fn pair_windows_match_reference_lengths_directions_strategies_and_errors() {
        for processor in 0..3 {
            let mut base = tokenizer(processor == 1);
            if processor == 2 {
                base.with_post_processor(Some(
                    TemplateProcessing::builder()
                        .try_single("[CLS] $A [SEP]")
                        .unwrap()
                        .try_pair("[CLS] $A [SEP] $B:3 [SEP]:3")
                        .unwrap()
                        .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
                        .build()
                        .unwrap(),
                ));
            }
            for direction in [TruncationDirection::Left, TruncationDirection::Right] {
                for strategy in [
                    TruncationStrategy::LongestFirst,
                    TruncationStrategy::OnlyFirst,
                    TruncationStrategy::OnlySecond,
                ] {
                    for max_length in [3, 4, 8, 17, 32] {
                        let mut reference = base.clone();
                        reference
                            .with_truncation(Some(TruncationParams {
                                max_length,
                                direction,
                                strategy,
                                ..Default::default()
                            }))
                            .unwrap();
                        let inference = InferenceTokenizer::new(reference.clone());
                        for left in [0, 1, 2, 7, 18] {
                            for right in [0, 1, 2, 7, 18] {
                                let a = "hello CAFÉ 東京 ! ".repeat(left);
                                let b = "world ! ".repeat(right);
                                let pair = (a.as_str(), b.as_str());
                                match (
                                    reference.encode(pair, true),
                                    inference.encode_batch(vec![pair]),
                                ) {
                                    (Ok(expected), Ok(actual)) => {
                                        equal_compact_inputs(
                                            &actual,
                                            std::slice::from_ref(&expected),
                                        );
                                        let InferenceInput::Original(actual) = &actual[0] else {
                                            panic!("expected pair")
                                        };
                                        assert_eq!(
                                            actual.get_attention_mask(),
                                            expected.get_attention_mask()
                                        );
                                        assert!(actual.get_overflowing().is_empty());
                                    }
                                    (Err(expected), Err(actual)) => {
                                        assert_eq!(actual.to_string(), expected.to_string())
                                    }
                                    _ => panic!(
                                        "pair success/error mismatch: {processor}/{direction:?}/{strategy:?}/{max_length}/{left}/{right}"
                                    ),
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn template_pair_truncation_preserves_inference_inputs() {
        for strategy in [
            TruncationStrategy::OnlyFirst,
            TruncationStrategy::OnlySecond,
        ] {
            let mut tokenizer = tokenizer(false);
            tokenizer.with_post_processor(Some(
                TemplateProcessing::builder()
                    .try_single("[CLS] $A [SEP]")
                    .unwrap()
                    .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")
                    .unwrap()
                    .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
                    .build()
                    .unwrap(),
            ));
            tokenizer
                .with_truncation(Some(TruncationParams {
                    max_length: 10,
                    strategy,
                    ..Default::default()
                }))
                .unwrap();
            let long = "hello world ".repeat(20);
            let pair = if strategy == TruncationStrategy::OnlyFirst {
                (long.as_str(), "CAFÉ!")
            } else {
                ("CAFÉ!", long.as_str())
            };
            equal_inputs(
                &encode_batch(&tokenizer, vec![pair]).unwrap(),
                &tokenizer.encode_batch(vec![pair], true).unwrap(),
            );
        }
    }

    fn equal_compact_inputs(actual: &[InferenceInput], expected: &[Encoding]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.get_ids(), expected.get_ids());
            assert_eq!(actual.get_type_ids(), expected.get_type_ids());
        }
    }

    #[test]
    fn guarded_fragments_preserve_controls_unicode_specials_and_word_decisions() {
        let mut texts = vec![
            String::new(),
            "HELLO worlds".into(),
            "hello café 東京!".into(),
            "hello\u{301} worlds".into(),
            "hello\0worlds".into(),
            "hello\u{b}worlds".into(),
            "hello\u{c}worlds".into(),
            "hello\u{a0}worlds".into(),
            "[MASK]hello[SEP]worlds".into(),
            "hel[UNK]lo".into(),
            "[MA\0SK]".into(),
            "helloq".into(),
            "hello café [MASK] worlds".repeat(50),
            "a".repeat(100),
            "a".repeat(101),
        ];
        for byte in 0..128u8 {
            let character = char::from(byte);
            texts.push(format!("hel{character}lo café [MASK] worlds"));
            texts.push(format!("hello {character}[MASK]café worlds"));
        }
        for count in [15, 16, 17, 32, 33] {
            for barrier in [" ", "\t", "\r\n", " \t  "] {
                let group = vec!["hello\u{301}[MASK]café"; count].join(barrier);
                texts.push(format!("{barrier}{group}{barrier}"));
                texts.push(format!("{group}{barrier}worlds{barrier}{group}"));
            }
        }
        let pool = kodamaembed_cpu::worker_pool("tokenize-test", 2).unwrap();
        for maximum in [2, 3, 4, 8, 32, 256] {
            let mut reference = crate::bert_wordpiece::tests::configured();
            reference
                .with_truncation(Some(TruncationParams {
                    max_length: maximum,
                    ..Default::default()
                }))
                .unwrap();
            let inference = InferenceTokenizer::new(reference.clone());
            assert!(inference.ascii.is_some());
            let inputs: Vec<_> = texts.iter().map(String::as_str).collect();
            let expected = reference.encode_batch(inputs.clone(), true).unwrap();
            assert!(
                expected
                    .iter()
                    .all(|e| e.get_attention_mask().iter().all(|&v| v == 1))
            );
            for (input, expected) in inputs.iter().zip(&expected) {
                let alone = inference.encode_batch_in(&pool, vec![*input]).unwrap();
                equal_compact_inputs(&alone, std::slice::from_ref(expected));
            }
            equal_compact_inputs(
                &inference.encode_batch_in(&pool, inputs).unwrap(),
                &expected,
            );
        }
    }

    #[test]
    fn guarded_tokenizer_keeps_pairs_and_pretokenized_inputs_general() {
        let reference = crate::bert_wordpiece::tests::configured();
        let inference = InferenceTokenizer::new(reference.clone());
        assert!(inference.ascii.is_some());
        let inputs: Vec<EncodeInput<'_>> = vec![
            ("hello café", "[MASK] worlds").into(),
            EncodeInput::Single(InputSequence::from(vec!["hello", "café", "worlds"])),
            EncodeInput::Dual(
                InputSequence::from(vec!["hello", "café"]),
                InputSequence::from(vec!["worlds"]),
            ),
        ];
        let expected = reference.encode_batch(inputs.clone(), true).unwrap();
        let actual = inference.encode_batch(inputs).unwrap();
        assert!(
            actual
                .iter()
                .all(|e| matches!(e, InferenceInput::Original(_)))
        );
        equal_compact_inputs(&actual, &expected);
    }

    #[test]
    fn unsupported_configurations_retain_batch_padding_and_pair_segment_types() {
        let mut padded = crate::bert_wordpiece::tests::configured();
        padded.with_padding(Some(tokenizers::PaddingParams::default()));
        let mut left = crate::bert_wordpiece::tests::configured();
        left.get_truncation_mut().unwrap().direction = TruncationDirection::Left;
        let mut runtime_specials = crate::bert_wordpiece::tests::configured();
        runtime_specials.set_encode_special_tokens(true);
        for reference in [padded, left, runtime_specials, tokenizer(false)] {
            let inference = InferenceTokenizer::new(reference.clone());
            assert!(inference.ascii.is_none());
            let inputs: Vec<EncodeInput<'_>> = vec![
                "hello".into(),
                "hello [MASK] café worlds hello worlds".into(),
                ("hello", "world").into(),
            ];
            let expected = reference.encode_batch(inputs.clone(), true).unwrap();
            let actual = inference.encode_batch(inputs).unwrap();
            equal_compact_inputs(&actual, &expected);
            for (actual, expected) in actual.iter().zip(expected) {
                let InferenceInput::Original(actual) = actual else {
                    panic!("expected reference encoding")
                };
                assert_eq!(actual.get_attention_mask(), expected.get_attention_mask());
            }
        }
    }
}
