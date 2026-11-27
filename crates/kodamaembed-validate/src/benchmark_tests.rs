use super::*;

const CORPUS: &str = r#"{"_id": "d1", "title": "Alpha", "text": "first"}
{"_id": "d2", "title": "", "text": "second"}
{"_id": "d3", "title": "Gamma", "text": "third"}
"#;
const QUERIES: &str = r#"{"_id": "q1", "text": "one"}
{"_id": "q2", "text": "two"}
{"_id": "q3", "text": "three"}
"#;

#[test]
fn beir_joins_titles_and_orders_queries_as_judged() {
    let judgments =
        "query-id\tcorpus-id\tscore\nq2\td3\t1\nq1\td1\t1\nq2\td2\t1\nq3\td2\t0\nq1\tmissing\t1\n";
    let task = beir(CORPUS, QUERIES, judgments).unwrap();
    assert_eq!(task.passages, ["Alpha first", "second", "Gamma third"]);
    // q3's only judgment scores 0, so q3 is not a test query.
    assert_eq!(task.queries, ["two", "one"]);
    assert_eq!(task.relevant, [vec![2, 1], vec![0]]);
    assert_eq!(task.unjudged, 1);
}

#[test]
fn beir_rejects_judgments_of_unknown_queries_and_malformed_lines() {
    let unknown = "query-id\tcorpus-id\tscore\nq9\td1\t1\n";
    assert!(beir(CORPUS, QUERIES, unknown).is_err());
    let short = "query-id\tcorpus-id\tscore\nq1\td1\n";
    assert!(beir(CORPUS, QUERIES, short).is_err());
}

#[test]
fn locomo_orders_sessions_by_number_and_keeps_judged_questions() {
    let json = r#"[{
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "A",
            "session_10_date_time": "later",
            "session_10": [{"speaker": "B", "dia_id": "D10:1", "text": "last"}],
            "session_2": [
                {"speaker": "A", "dia_id": "D2:1", "text": "hello"},
                {"speaker": "B", "dia_id": "D2:2", "text": "look", "blip_caption": "a dog"}
            ]
        },
        "qa": [
            {"question": "Who said last?", "evidence": ["D10:1"], "category": 1},
            {"question": "What did B show?", "evidence": ["D2:2", "D9:9"], "category": 1},
            {"question": "Unanswerable?", "evidence": ["D7:1"], "category": 5},
            {"question": "No evidence?", "category": 5}
        ]
    }]"#;
    let tasks = locomo(json.as_bytes()).unwrap();
    assert_eq!(tasks.len(), 1);
    let task = &tasks[0];
    assert_eq!(task.name, "conv-1");
    assert_eq!(
        task.passages,
        ["A: hello", "B: look (shares a photo: a dog)", "B: last"]
    );
    assert_eq!(task.queries, ["Who said last?", "What did B show?"]);
    assert_eq!(task.relevant, [vec![2], vec![1]]);
    assert_eq!(task.unjudged, 2);
}

fn unit(values: &[[f32; 2]]) -> Matrix {
    Matrix::from_rows(&values.iter().map(|v| v.to_vec()).collect::<Vec<_>>()).unwrap()
}

#[test]
fn scores_follow_the_ranking() {
    // Passage scores for the query [1, 0]: 0.9, 0.1, 0.5 -> ranking 0, 2, 1.
    let passages = unit(&[[0.9, 0.0], [0.1, 0.0], [0.5, 0.0]]);
    let queries = unit(&[[1.0, 0.0], [1.0, 0.0]]);
    // Query 0 wants passage 2 (rank 1); query 1 wants passages 0 and 1 (ranks 0 and 2).
    let scores = score(&passages, &queries, &[vec![2], vec![0, 1]]).unwrap();
    let gain = |rank: f64| 1.0 / (rank + 2.0).log2();
    let first = gain(1.0) / gain(0.0);
    let second = (gain(0.0) + gain(2.0)) / (gain(0.0) + gain(1.0));
    assert_eq!(scores.queries, 2);
    assert!((scores.ndcg_at_10 - (first + second) / 2.0).abs() < 1e-12);
    assert!((scores.recall_at_10 - 1.0).abs() < 1e-12);
    assert!((scores.hit_at_10 - 1.0).abs() < 1e-12);
}

#[test]
fn a_relevant_passage_below_rank_10_earns_nothing() {
    let rows: Vec<[f32; 2]> = (0..12).map(|i| [1.0 - i as f32 * 0.05, 0.0]).collect();
    let passages = unit(&rows);
    let queries = unit(&[[1.0, 0.0]]);
    let scores = score(&passages, &queries, &[vec![11]]).unwrap();
    assert_eq!(scores.ndcg_at_10, 0.0);
    assert_eq!(scores.recall_at_10, 0.0);
    assert_eq!(scores.hit_at_10, 0.0);
}

#[test]
fn combining_weights_tasks_by_their_queries() {
    let a = Scores {
        queries: 1,
        ndcg_at_10: 1.0,
        recall_at_10: 1.0,
        hit_at_10: 1.0,
    };
    let b = Scores {
        queries: 3,
        ndcg_at_10: 0.0,
        recall_at_10: 0.0,
        hit_at_10: 0.0,
    };
    let total = combine(&[a, b]);
    assert_eq!(total.queries, 4);
    assert!((total.ndcg_at_10 - 0.25).abs() < 1e-12);
}

#[test]
fn identical_top_10_counts_queries_whose_sets_agree() {
    let rows: Vec<[f32; 2]> = (0..12).map(|i| [1.0 - i as f32 * 0.05, 0.0]).collect();
    let passages = unit(&rows);
    let mut moved = rows.clone();
    moved[11] = [2.0, 0.0];
    let moved = unit(&moved);
    let queries = unit(&[[1.0, 0.0], [0.0, 1.0]]);
    assert_eq!(
        identical_top_10((&passages, &queries), (&passages, &queries)).unwrap(),
        2
    );
    // Passage 11 jumps into the top 10 for the first query only (the second
    // query scores every passage 0 and ranks by index either way).
    assert_eq!(
        identical_top_10((&passages, &queries), (&moved, &queries)).unwrap(),
        1
    );
}

fn conversation(name: &str) -> serde_json::Value {
    serde_json::json!({
        "sample_id": name,
        "conversation": {"session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "one"}]},
        "qa": [{"question": "which?", "evidence": ["D1:1"]}]
    })
}

#[test]
fn locomo_preserves_portable_task_ids_and_rejects_unsafe_names() {
    for name in ["conv-1", "26c7ad4b-5ab5-4a68-9a15-215674c115a0", "A_2.v1"] {
        let bytes = serde_json::to_vec(&vec![conversation(name)]).unwrap();
        assert_eq!(locomo(&bytes).unwrap()[0].name, name);
    }
    for name in [
        "",
        ".",
        "..",
        "../escape",
        "a/b",
        "a\\b",
        "C:escape",
        "/absolute",
        "\\\\server\\share",
        "a:stream",
        "a.",
        "a ",
        "a*",
        "é",
        "CON",
        "con.txt",
        "PrN",
        "AUX.bin",
        "nul",
        "COM1",
        "lpt9.data",
    ] {
        let bytes = serde_json::to_vec(&vec![conversation(name)]).unwrap();
        assert!(locomo(&bytes).is_err(), "accepted {name:?}");
    }
    let bytes = serde_json::to_vec(&vec![conversation(&"a".repeat(129))]).unwrap();
    assert!(locomo(&bytes).is_err());
}

#[test]
fn locomo_rejects_same_and_case_alias_task_ids_before_output_names_can_collide() {
    for names in [["conv-1", "conv-1"], ["Conv-1", "conv-1"]] {
        let bytes = serde_json::to_vec(&names.map(conversation)).unwrap();
        assert!(locomo(&bytes).is_err());
    }
}

#[test]
fn loaders_reject_ambiguous_document_query_turn_and_positive_judgment_ids() {
    let judgments = "query-id\tcorpus-id\tscore\nq1\td1\t1\n";
    assert!(beir(&format!("{CORPUS}{CORPUS}"), QUERIES, judgments).is_err());
    assert!(beir(CORPUS, &format!("{QUERIES}{QUERIES}"), judgments).is_err());
    let repeated = "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td1\t2\n";
    assert!(beir(CORPUS, QUERIES, repeated).is_err());
    let mut duplicated = conversation("conv-1");
    duplicated["conversation"]["session_2"] = duplicated["conversation"]["session_1"].clone();
    assert!(locomo(&serde_json::to_vec(&vec![duplicated]).unwrap()).is_err());
}

#[test]
fn locomo_stably_deduplicates_mapped_evidence_and_preserves_missing_interleave() {
    let mut sample = conversation("conv-1");
    sample["conversation"]["session_1"] = serde_json::json!([
        {"speaker": "A", "dia_id": "D1:1", "text": "one"},
        {"speaker": "B", "dia_id": "D1:2", "text": "two"},
        {"speaker": "A", "dia_id": "D1:3", "text": "three"}
    ]);
    sample["qa"] = serde_json::json!([
        {"question": "which?", "evidence": ["missing", "D1:2", "D1:2", "missing", "D1:1", "D1:2"]},
        {"question": "still unjudged?", "evidence": ["missing", "missing"]},
        {"question": "another?", "evidence": ["D1:3", "D1:3"]}
    ]);
    let tasks = locomo(&serde_json::to_vec(&vec![sample]).unwrap()).unwrap();
    assert_eq!(tasks[0].passages, ["A: one", "B: two", "A: three"]);
    assert_eq!(tasks[0].queries, ["which?", "another?"]);
    assert_eq!(tasks[0].relevant, [vec![1, 0], vec![2]]);
    assert_eq!(tasks[0].unjudged, 1);
}

#[test]
fn deduplicated_binary_evidence_uses_distinct_recall_and_ideal_dcg() {
    let mut sample = conversation("conv-1");
    sample["conversation"]["session_1"] = serde_json::json!([
        {"speaker": "A", "dia_id": "D1:1", "text": "one"},
        {"speaker": "B", "dia_id": "D1:2", "text": "two"}
    ]);
    sample["qa"][0]["evidence"] = serde_json::json!(["D1:1", "D1:1", "D1:2"]);
    let task = locomo(&serde_json::to_vec(&vec![sample]).unwrap())
        .unwrap()
        .remove(0);
    let passages = unit(&[[1.0, 0.0], [0.5, 0.0]]);
    let queries = unit(&[[1.0, 0.0]]);
    let scores = score(&passages, &queries, &task.relevant).unwrap();
    assert_eq!(scores.recall_at_10, 1.0);
    assert_eq!(scores.ndcg_at_10, 1.0);
    assert_eq!(scores.hit_at_10, 1.0);
    // A partial hit detects an IDCG denominator that counts repeated citations.
    let mut rows = vec![[0.5, 0.0]; 11];
    rows[0] = [1.0, 0.0];
    rows[10] = [0.0, 0.0];
    let partial = score(&unit(&rows), &queries, &[vec![0, 10]]).unwrap();
    assert_eq!(partial.recall_at_10, 0.5);
    assert_eq!(partial.ndcg_at_10, 1.0 / (1.0 + 1.0 / 3.0_f64.log2()));
    assert!(score(&passages, &queries, &[vec![0, 0, 1]]).is_err());
    assert_eq!(RETRIEVAL_METRIC_CONVENTION, "binary-distinct-relevance-v1");
}

#[test]
fn retrieval_rejects_nonfinite_values_while_generic_fidelity_still_counts_them() {
    let finite = unit(&[[1.0, 0.0]]);
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let invalid = unit(&[[value, 0.0]]);
        assert!(score(&invalid, &finite, &[vec![0]]).is_err());
        assert!(score(&finite, &invalid, &[vec![0]]).is_err());
        assert!(identical_top_10((&finite, &finite), (&invalid, &finite)).is_err());
        assert!(identical_top_10((&finite, &invalid), (&finite, &finite)).is_err());
        assert!(compare((&finite, &finite), (&invalid, &finite), &[vec![0]]).is_err());
        assert!(compare((&finite, &invalid), (&finite, &finite), &[vec![0]]).is_err());
        assert_eq!(
            crate::fidelity::compare(&finite, &invalid)
                .unwrap()
                .non_finite,
            1
        );
    }
}

#[test]
fn retrieval_rejects_inconsistent_shapes_and_judgments_before_ranking() {
    let finite = unit(&[[1.0, 0.0]]);
    for invalid in [
        Matrix {
            rows: 2,
            dim: 2,
            data: vec![1.0],
        },
        Matrix {
            rows: usize::MAX,
            dim: 2,
            data: vec![],
        },
        Matrix {
            rows: 1,
            dim: 0,
            data: vec![],
        },
    ] {
        assert!(score(&invalid, &finite, &[vec![0]]).is_err());
        assert!(score(&finite, &invalid, &[vec![0]]).is_err());
        assert!(identical_top_10((&finite, &finite), (&invalid, &finite)).is_err());
        assert!(compare((&finite, &finite), (&invalid, &finite), &[vec![0]]).is_err());
        assert!(compare((&finite, &finite), (&finite, &invalid), &[vec![0]]).is_err());
    }
    let other_dim = Matrix::from_rows(&[vec![1.0]]).unwrap();
    assert!(score(&finite, &other_dim, &[vec![0]]).is_err());
    for relevant in [
        vec![],
        vec![vec![]],
        vec![vec![1]],
        vec![vec![0, 0]],
        vec![vec![0], vec![0]],
    ] {
        assert!(score(&finite, &finite, &relevant).is_err());
        assert!(compare((&finite, &finite), (&finite, &finite), &relevant).is_err());
    }
    let two = unit(&[[1.0, 0.0], [0.0, 1.0]]);
    assert!(identical_top_10((&finite, &finite), (&two, &finite)).is_err());
    assert!(identical_top_10((&finite, &finite), (&finite, &two)).is_err());
    assert!(identical_top_10((&finite, &finite), (&other_dim, &other_dim)).is_err());
    assert!(compare((&finite, &finite), (&other_dim, &other_dim), &[vec![0]]).is_err());
}

#[test]
fn retrieval_with_no_judged_queries_keeps_zero_metrics() {
    let empty = Matrix::from_rows(&[]).unwrap();
    let passages = unit(&[[1.0, 0.0]]);
    assert_eq!(score(&passages, &empty, &[]).unwrap(), Scores::default());
    assert_eq!(score(&empty, &empty, &[]).unwrap(), Scores::default());
    assert_eq!(
        identical_top_10((&passages, &empty), (&passages, &empty)).unwrap(),
        0
    );
    assert!(score(&empty, &passages, &[vec![0]]).is_err());
    let compared = compare((&passages, &empty), (&passages, &empty), &[]).unwrap();
    assert_eq!(compared.reference, Scores::default());
    assert_eq!(compared.mean_top10_overlap, 0.0);
    let gates = quality_gates(&compared, QualityFidelity::default());
    assert!(!gates.evaluated);
    assert!(!gates.pass);
    assert!(compare((&empty, &passages), (&empty, &passages), &[vec![0]]).is_err());
}

#[test]
fn shared_rankings_distinguish_mean_overlap_from_identical_set_fraction() {
    let rows: Vec<[f32; 2]> = (0..12).map(|i| [1.0 - i as f32 * 0.05, 0.0]).collect();
    let reference = unit(&rows);
    let mut moved = rows;
    moved[11] = [2.0, 0.0];
    let candidate = unit(&moved);
    let queries = unit(&[[1.0, 0.0], [0.0, 1.0]]);
    let relevant = [vec![0, 11], vec![2]];
    let compared = compare((&reference, &queries), (&candidate, &queries), &relevant).unwrap();
    assert_eq!(compared.identical_top10_sets, 1);
    assert_eq!(compared.mean_top10_overlap, (0.9 + 1.0) / 2.0);
    assert_ne!(compared.mean_top10_overlap, 1.0 / 2.0);
    assert_eq!(
        compared.reference,
        score(&reference, &queries, &relevant).unwrap()
    );
    assert_eq!(
        compared.candidate,
        score(&candidate, &queries, &relevant).unwrap()
    );
    assert_eq!(
        compared.identical_top10_sets,
        identical_top_10((&reference, &queries), (&candidate, &queries)).unwrap()
    );
    // With fewer than ten passages, overlap uses the actual set size.
    let small = unit(&[[1.0, 0.0], [0.0, 1.0]]);
    let reversed = unit(&[[0.0, 1.0], [1.0, 0.0]]);
    let one = unit(&[[1.0, 0.0]]);
    let compared = compare((&small, &one), (&reversed, &one), &[vec![0]]).unwrap();
    assert_eq!(compared.mean_top10_overlap, 1.0);
    assert_eq!(compared.identical_top10_sets, 1);
}

#[test]
fn shared_score_fold_matches_independent_arithmetic_and_tie_policy() {
    // Scored a second way, so ties, cutoff misses and several judgments cannot share a bug.
    fn independent(passages: &Matrix, queries: &Matrix, relevant: &[Vec<usize>]) -> Scores {
        let mut totals = Scores {
            queries: relevant.len(),
            ..Scores::default()
        };
        let gain = |rank: usize| 1.0 / ((rank + 2) as f64).log2();
        for (query, wanted) in relevant.iter().enumerate() {
            let order = ranking::ranked(&ranking::scores(passages, queries.row(query)));
            let found: Vec<usize> = (0..K.min(order.len()))
                .filter(|&rank| wanted.contains(&order[rank]))
                .collect();
            let ideal: f64 = (0..wanted.len().min(K)).map(gain).sum();
            totals.ndcg_at_10 += found.iter().map(|&rank| gain(rank)).sum::<f64>() / ideal;
            totals.recall_at_10 += found.len() as f64 / wanted.len() as f64;
            if !found.is_empty() {
                totals.hit_at_10 += 1.0;
            }
        }
        let n = relevant.len().max(1) as f64;
        totals.ndcg_at_10 /= n;
        totals.recall_at_10 /= n;
        totals.hit_at_10 /= n;
        totals
    }
    let rows: Vec<_> = (0..14)
        .map(|i| [i as f32 * 0.25 - 1.0, (i % 3) as f32 - 1.0])
        .collect();
    let passages = unit(&rows);
    let queries = unit(&[[1.0, 0.0], [0.0, 0.0], [-1.0, 0.5], [1.0, -1.0]]);
    let relevant = [vec![0], vec![0, 9, 13], vec![3, 6], vec![2, 4, 8, 12]];
    let compared = compare((&passages, &queries), (&passages, &queries), &relevant).unwrap();
    let expected = independent(&passages, &queries, &relevant);
    for got in [
        score(&passages, &queries, &relevant).unwrap(),
        compared.reference,
        compared.candidate,
    ] {
        assert_eq!(got.ndcg_at_10.to_bits(), expected.ndcg_at_10.to_bits());
        assert_eq!(got.recall_at_10.to_bits(), expected.recall_at_10.to_bits());
        assert_eq!(got.hit_at_10.to_bits(), expected.hit_at_10.to_bits());
    }
    let zeros = ranking::scores(&passages, queries.row(1));
    assert_eq!(zeros[0].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(zeros[1].to_bits(), 0.0_f64.to_bits());
    // Preserve total_cmp's distinction between signed zeros, then index ties.
    assert_eq!(
        ranking::ranked(&zeros),
        [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 3]
    );
}

#[test]
fn comparison_aggregation_weights_overlap_by_queries_and_preserves_scores() {
    let a = Comparison {
        reference: Scores {
            queries: 2,
            ndcg_at_10: 0.4,
            recall_at_10: 0.5,
            hit_at_10: 1.0,
        },
        candidate: Scores {
            queries: 2,
            ndcg_at_10: 0.3,
            recall_at_10: 0.4,
            hit_at_10: 0.5,
        },
        identical_top10_sets: 1,
        mean_top10_overlap: 0.95,
        max_unit_norm_error: 1e-6,
    };
    let b = Comparison {
        reference: Scores {
            queries: 1,
            ..Scores::default()
        },
        candidate: Scores {
            queries: 1,
            ..Scores::default()
        },
        mean_top10_overlap: 0.5,
        max_unit_norm_error: 2e-6,
        ..Comparison::default()
    };
    let result = combine_comparisons(&[a.clone(), b.clone()]);
    assert_eq!(result.mean_top10_overlap, (0.95 * 2.0 + 0.5) / 3.0);
    assert_eq!(result.identical_top10_sets, 1);
    assert_eq!(result.max_unit_norm_error, 2e-6);
    assert_eq!(result.reference, combine(&[a.reference, b.reference]));
    assert_eq!(result.candidate, combine(&[a.candidate, b.candidate]));
}

fn boundary_quality() -> (Comparison, QualityFidelity) {
    (
        Comparison {
            reference: Scores {
                queries: 1,
                ndcg_at_10: 0.005,
                recall_at_10: 0.01,
                hit_at_10: 0.01,
            },
            candidate: Scores {
                queries: 1,
                ..Scores::default()
            },
            mean_top10_overlap: 0.9,
            max_unit_norm_error: 1e-5,
            ..Comparison::default()
        },
        QualityFidelity {
            mean_passage_cosine_distance: 1e-3,
            mean_query_cosine_distance: 1e-3,
            max_l2: 0.1,
            non_finite: 0,
        },
    )
}

#[test]
fn quality_gates_include_boundaries_and_reject_the_next_float_outside_each_limit() {
    let (comparison, fidelity) = boundary_quality();
    assert!(quality_gates(&comparison, fidelity).pass);
    for field in 0..9 {
        let (mut c, mut f) = boundary_quality();
        match field {
            0 => f.mean_passage_cosine_distance = 1e-3_f64.next_up(),
            1 => f.mean_query_cosine_distance = 1e-3_f64.next_up(),
            2 => f.max_l2 = 0.1_f64.next_up(),
            3 => c.max_unit_norm_error = 1e-5_f64.next_up(),
            4 => f.non_finite = 1,
            5 => c.mean_top10_overlap = 0.9_f64.next_down(),
            6 => c.reference.ndcg_at_10 = 0.005_f64.next_up(),
            7 => c.reference.recall_at_10 = 0.01_f64.next_up(),
            8 => c.reference.hit_at_10 = 0.01_f64.next_up(),
            _ => unreachable!(),
        }
        assert!(!quality_gates(&c, f).pass, "outside gate {field} passed");
    }
}

#[test]
fn quality_gates_never_accept_nonfinite_statistics_or_mismatched_query_counts() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for field in 0..8 {
            let (mut c, mut f) = boundary_quality();
            match field {
                0 => f.mean_passage_cosine_distance = value,
                1 => f.mean_query_cosine_distance = value,
                2 => f.max_l2 = value,
                3 => c.max_unit_norm_error = value,
                4 => c.mean_top10_overlap = value,
                5 => c.reference.ndcg_at_10 = value,
                6 => c.reference.recall_at_10 = value,
                7 => c.reference.hit_at_10 = value,
                _ => unreachable!(),
            }
            assert!(!quality_gates(&c, f).pass);
        }
    }
    let (mut comparison, fidelity) = boundary_quality();
    comparison.candidate.queries = 2;
    let gates = quality_gates(&comparison, fidelity);
    assert!(!gates.evaluated);
    assert!(!gates.pass);
}

#[test]
fn comparison_checks_absolute_unit_norms_including_the_reference() {
    let queries = unit(&[[1.0, 0.0]]);
    let passages = unit(&[[0.0, 1.0]]);
    let compared = compare((&passages, &queries), (&passages, &queries), &[vec![0]]).unwrap();
    assert_eq!(compared.max_unit_norm_error, 0.0);
    assert!(quality_gates(&compared, QualityFidelity::default()).pass);
    let invalid_norm = unit(&[[2.0, 0.0]]);
    for pair in [
        ((&invalid_norm, &queries), (&passages, &queries)),
        ((&passages, &queries), (&invalid_norm, &queries)),
    ] {
        let compared = compare(pair.0, pair.1, &[vec![0]]).unwrap();
        assert_eq!(compared.max_unit_norm_error, 1.0);
        assert!(!quality_gates(&compared, QualityFidelity::default()).fidelity_pass);
    }
}
