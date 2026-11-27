use super::{Capacity, MAX_BATCH_SEQUENCES, MAX_BATCH_TOKENS, STREAM_TILE_ROWS};

pub(super) const INPUTS: usize = 10;

// This order is the packed host/device input contract.
#[derive(Clone, Copy)]
#[repr(usize)]
pub(super) enum InputSection {
    Ids,
    Types,
    Positions,
    SequenceCodes,
    RowGroups,
    SharedRows,
    SequenceStarts,
    SequenceLengths,
    PaddedStarts,
    Classes,
}

pub(super) struct InputSections<'a> {
    pub ids: &'a [i32],
    pub types: &'a [i32],
    pub positions: &'a [i32],
    pub sequence_codes: &'a [i32],
    pub row_groups: &'a [i32],
    pub shared_rows: &'a [i32],
    pub sequence_starts: &'a [i32],
    pub sequence_lengths: &'a [i32],
    pub padded_starts: &'a [i32],
    pub classes: &'a [i32],
}

impl<'a> InputSections<'a> {
    pub(super) fn as_array(&self) -> [&'a [i32]; INPUTS] {
        [
            self.ids,
            self.types,
            self.positions,
            self.sequence_codes,
            self.row_groups,
            self.shared_rows,
            self.sequence_starts,
            self.sequence_lengths,
            self.padded_starts,
            self.classes,
        ]
    }
}

#[derive(Clone, Copy)]
pub(super) struct InputLayout {
    lengths: [usize; INPUTS],
    offsets: [usize; INPUTS],
    words: usize,
}

impl InputLayout {
    pub(super) fn aligned_words(len: usize) -> usize {
        len.next_multiple_of(64)
    }

    pub(super) fn new(lengths: [usize; INPUTS]) -> Self {
        let mut offsets = [0; INPUTS];
        let mut words = 0;
        for (offset, len) in offsets.iter_mut().zip(lengths) {
            *offset = words;
            // Each view and every vectorized staging copy starts aligned.
            words += Self::aligned_words(len);
        }
        Self {
            lengths,
            offsets,
            words,
        }
    }

    pub(super) fn for_capacity(capacity: Capacity) -> Self {
        let (t, s, g) = (capacity.tokens, capacity.sequences, capacity.groups);
        Self::new([t, t, t, t, t, t, s, s, s, g])
    }

    pub(super) fn words(self) -> usize {
        self.words
    }

    pub(super) fn len(self, section: InputSection) -> usize {
        self.lengths[section as usize]
    }

    pub(super) fn byte_offset(self, section: InputSection) -> usize {
        self.offsets[section as usize] * size_of::<i32>()
    }

    pub(super) fn pack(self, target: &mut [i32], sections: [&[i32]; INPUTS]) {
        debug_assert_eq!(self.lengths, sections.map(<[i32]>::len));
        let packed = &mut target[..self.words];
        packed.fill(0);
        for (&offset, values) in self.offsets.iter().zip(sections) {
            packed[offset..offset + values.len()].copy_from_slice(values);
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Region {
    pub offset: usize,
    pub bytes: usize,
}

impl Region {
    pub(super) fn append_values<T>(end: &mut usize, count: usize) -> Self {
        let offset = end.next_multiple_of(256);
        let bytes = count * size_of::<T>();
        *end = offset + bytes;
        Self { offset, bytes }
    }

    fn end(self) -> usize {
        self.offset + self.bytes
    }
}

pub(super) struct ArenaLayout {
    pub pre: Region,
    pub queries: Region,
    pub keys: Region,
    pub values16: Region,
    pub value_planes: Region,
    pub intermediate: Region,
    pub bytes: usize,
}

impl ArenaLayout {
    pub(super) fn new(tokens: usize, hidden: usize, intermediate: usize, padded: usize) -> Self {
        let mut end = 0;
        let mut append = |bytes| Region::append_values::<u8>(&mut end, bytes);
        let pre = append(tokens * hidden * 2);
        let queries = append(tokens * hidden);
        let keys = append(tokens * hidden);
        let values16 = append(tokens * hidden * 2);
        let value_planes = append(hidden * padded);
        // The stream's normalization readers retain this prefix during up.
        let intermediate = Region {
            offset: (tokens.min(STREAM_TILE_ROWS) * hidden * 2).next_multiple_of(256),
            bytes: tokens * intermediate * 2,
        };
        Self {
            pre,
            queries,
            keys,
            values16,
            value_planes,
            intermediate,
            bytes: value_planes.end().max(intermediate.end()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BatchLengths {
    pub tokens: usize,
    sequences: usize,
}

impl BatchLengths {
    pub(super) fn new(lengths: &[usize], max_length: usize) -> Option<Self> {
        if lengths.is_empty() || lengths.iter().any(|&len| len == 0 || len > max_length) {
            return None;
        }
        let tokens = lengths
            .iter()
            .try_fold(0usize, |sum, &len| sum.checked_add(len))?;
        Some(Self {
            tokens,
            sequences: lengths.len(),
        })
    }

    pub(super) fn check_grids(self) -> Result<(), &'static str> {
        if self.tokens > MAX_BATCH_TOKENS {
            return Err("more tokens than the tall projections' launch grids hold");
        }
        if self.sequences > MAX_BATCH_SEQUENCES {
            return Err("more sequences than the wide attention's launch grids hold");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_inputs_keep_the_packed_contract_and_zero_padding() {
        let payloads: [Vec<i32>; INPUTS] = std::array::from_fn(|i| vec![i as i32 + 1; i + 1]);
        let sections = InputSections {
            ids: &payloads[0],
            types: &payloads[1],
            positions: &payloads[2],
            sequence_codes: &payloads[3],
            row_groups: &payloads[4],
            shared_rows: &payloads[5],
            sequence_starts: &payloads[6],
            sequence_lengths: &payloads[7],
            padded_starts: &payloads[8],
            classes: &payloads[9],
        };
        let named = [
            InputSection::Ids,
            InputSection::Types,
            InputSection::Positions,
            InputSection::SequenceCodes,
            InputSection::RowGroups,
            InputSection::SharedRows,
            InputSection::SequenceStarts,
            InputSection::SequenceLengths,
            InputSection::PaddedStarts,
            InputSection::Classes,
        ];
        let layout = InputLayout::new(sections.as_array().map(<[i32]>::len));
        let mut packed = vec![-1; layout.words() + 3];
        layout.pack(&mut packed, sections.as_array());
        for (index, section) in named.into_iter().enumerate() {
            assert_eq!(layout.byte_offset(section), index * 256);
            assert_eq!(layout.len(section), index + 1);
            let start = layout.byte_offset(section) / 4;
            assert_eq!(&packed[start..start + index + 1], &payloads[index]);
            assert!(
                packed[start + index + 1..start + 64]
                    .iter()
                    .all(|&v| v == 0)
            );
        }
        assert_eq!(&packed[layout.words()..], &[-1; 3]);
    }

    #[test]
    fn input_layout_fits_componentwise_capacity_growth() {
        let old = Capacity {
            tokens: 65,
            sequences: 3,
            groups: 9,
            padded: 192,
        };
        let need = Capacity {
            tokens: 129,
            sequences: 2,
            groups: 6,
            padded: 256,
        };
        let grown = old.grow_to(need);
        assert!(grown.covers(old) && grown.covers(need));
        assert_eq!(grown.tokens, 144);
        assert_eq!(grown.grow_to(old).tokens, 144);
        let large = Capacity {
            tokens: 513,
            ..need
        };
        assert_eq!(grown.grow_to(large).tokens, large.tokens);
        let capacity = InputLayout::for_capacity(grown);
        for batch in [old, need, grown] {
            let layout = InputLayout::for_capacity(batch);
            assert!(layout.words() <= capacity.words());
            for i in 0..INPUTS {
                assert_eq!(layout.offsets[i] % 64, 0);
                assert!(layout.offsets[i] + layout.lengths[i] <= layout.words());
            }
        }
        assert_eq!(capacity.words(), 6 * 192 + 4 * 64);
    }

    #[test]
    fn independent_rows_start_aligned_after_the_live_arena_tail() {
        let mut end = 513;
        let half_row = Region::append_values::<u16>(&mut end, 65);
        let steps = Region::append_values::<f32>(&mut end, 3);
        assert_eq!((half_row.offset, half_row.bytes), (768, 130));
        assert_eq!((steps.offset, steps.bytes, end), (1024, 12, 1036));
    }

    #[test]
    fn arena_offsets_preserve_the_existing_device_views() {
        for (tokens, hidden, padded, offsets, bytes) in [
            (1, 384, 64, [0, 768, 1280, 1792, 2560, 768], 27_136),
            (
                65,
                1024,
                128,
                [0, 133_120, 199_680, 266_240, 399_360, 32_768],
                565_248,
            ),
        ] {
            let layout = ArenaLayout::new(tokens, hidden, 4 * hidden, padded);
            assert_eq!(
                [
                    layout.pre.offset,
                    layout.queries.offset,
                    layout.keys.offset,
                    layout.values16.offset,
                    layout.value_planes.offset,
                    layout.intermediate.offset
                ],
                offsets,
            );
            assert_eq!(layout.bytes, bytes);
        }
    }

    #[test]
    fn arena_views_keep_alignment_bounds_and_live_stream_prefix() {
        for hidden in [384, 1024] {
            for tokens in [
                1,
                2,
                16,
                17,
                31,
                32,
                33,
                64,
                65,
                128,
                129,
                512,
                1024,
                MAX_BATCH_TOKENS,
            ] {
                let padded = tokens.next_multiple_of(64);
                let layout = ArenaLayout::new(tokens, hidden, 4 * hidden, padded);
                let attention = [
                    layout.pre,
                    layout.queries,
                    layout.keys,
                    layout.values16,
                    layout.value_planes,
                ];
                let mut end: usize = 0;
                for region in attention {
                    assert_eq!(region.offset, end.next_multiple_of(256));
                    assert_eq!(region.offset % 256, 0);
                    assert!(region.end() <= layout.bytes);
                    end = region.end();
                }
                assert_eq!(layout.intermediate.offset % 256, 0);
                assert_eq!(layout.intermediate.bytes, tokens * hidden * 8);
                assert!(layout.intermediate.offset >= tokens.min(STREAM_TILE_ROWS) * hidden * 2);
                assert!(layout.intermediate.end() <= layout.bytes);
                assert_eq!(layout.bytes, end.max(layout.intermediate.end()));
            }
        }
    }

    #[test]
    fn lengths_reject_empty_zero_overflow_and_model_limits() {
        for lengths in [&[][..], &[0], &[1, 0], &[usize::MAX, 1]] {
            assert!(BatchLengths::new(lengths, usize::MAX).is_none());
        }
        assert!(BatchLengths::new(&[129], 128).is_none());
        assert_eq!(BatchLengths::new(&[129], 512).unwrap().tokens, 129);
        assert_eq!(BatchLengths::new(&[1, 2, 512], 512).unwrap().tokens, 515);
    }

    #[test]
    fn length_grid_limits_keep_their_boundaries_and_error_order() {
        let at_tokens = BatchLengths::new(&vec![64; MAX_BATCH_SEQUENCES], 512).unwrap();
        assert_eq!(at_tokens.tokens, MAX_BATCH_TOKENS);
        assert!(at_tokens.check_grids().is_ok());
        let too_many_tokens = BatchLengths::new(&vec![65; MAX_BATCH_SEQUENCES], 512).unwrap();
        assert_eq!(
            too_many_tokens.check_grids(),
            Err("more tokens than the tall projections' launch grids hold")
        );
        let too_many_sequences = BatchLengths::new(&vec![1; MAX_BATCH_SEQUENCES + 1], 512).unwrap();
        assert_eq!(
            too_many_sequences.check_grids(),
            Err("more sequences than the wide attention's launch grids hold")
        );
        let both = BatchLengths::new(&vec![65; MAX_BATCH_SEQUENCES + 1], 512).unwrap();
        assert_eq!(both.check_grids(), too_many_tokens.check_grids());
    }
}
