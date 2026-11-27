use super::*;
use std::collections::VecDeque;

struct MockEngine {
    passage_results: VecDeque<Vec<Vec<f32>>>,
    query_results: VecDeque<Vec<Vec<f32>>>,
    query_calls: Vec<Vec<String>>,
}

impl MockEngine {
    fn new(passages: Vec<Vec<Vec<f32>>>, queries: Vec<Vec<Vec<f32>>>) -> Self {
        Self {
            passage_results: passages.into(),
            query_results: queries.into(),
            query_calls: Vec::new(),
        }
    }
}

impl Engine for MockEngine {
    fn embed_passages(&mut self, _: &[&str]) -> crate::Result<Vec<Vec<f32>>> {
        self.passage_results
            .pop_front()
            .ok_or_else(|| "unexpected passage call".into())
    }

    fn embed_queries(&mut self, texts: &[&str]) -> crate::Result<Vec<Vec<f32>>> {
        self.query_calls
            .push(texts.iter().map(|text| text.to_string()).collect());
        self.query_results
            .pop_front()
            .ok_or_else(|| "unexpected query call".into())
    }

    fn report(&self) -> Option<serde_json::Value> {
        None
    }
}

fn task() -> Task {
    Task {
        name: "task-1".into(),
        passages: vec!["a".into(), "b".into()],
        queries: vec!["q".into()],
        relevant: vec![vec![0]],
        unjudged: 0,
    }
}

#[test]
fn each_family_is_chosen_by_name_and_takes_its_models_text_settings() {
    use clap::ValueEnum;
    let settings = |name: &str| {
        let spec = Family::from_str(name, false).unwrap().spec();
        (spec.query_prefix, spec.passage_prefix, spec.max_length)
    };
    assert_eq!(settings("e5"), ("query: ", "passage: ", 512));
    assert_eq!(settings("minilm"), ("", "", 256));
    assert_eq!(
        settings("modernbert-embed"),
        ("search_query: ", "search_document: ", 8192)
    );
    assert!(Family::from_str("bert", false).is_err());
}

#[test]
fn retrieval_task_checks_both_counts_and_can_reuse_engine_after_rejection() {
    for bad in [vec![vec![1.0, 0.0]], vec![vec![1.0, 0.0]; 3]] {
        let mut engine = MockEngine::new(vec![bad], Vec::new());
        assert!(embed_task(&mut engine, &task()).is_err());
        assert!(engine.query_calls.is_empty());
    }
    for bad in [vec![], vec![vec![1.0, 0.0]; 2]] {
        let mut engine = MockEngine::new(vec![vec![vec![1.0, 0.0]; 2]], vec![bad]);
        assert!(embed_task(&mut engine, &task()).is_err());
    }
    let mut engine = MockEngine::new(
        vec![vec![], vec![vec![1.0, 0.0], vec![2.0, 0.0]]],
        vec![vec![vec![3.0, 0.0]]],
    );
    assert!(embed_task(&mut engine, &task()).is_err());
    let (passages, queries) = embed_task(&mut engine, &task()).unwrap();
    assert_eq!(passages.row(1), &[2.0, 0.0]);
    assert_eq!(queries.row(0), &[3.0, 0.0]);
    assert_eq!(engine.query_calls.len(), 1);
}

#[test]
fn retrieval_task_with_no_queries_accepts_an_empty_backend_result() {
    let mut task = task();
    task.queries.clear();
    task.relevant.clear();
    let mut engine = MockEngine::new(vec![vec![vec![1.0, 0.0]; 2]], vec![vec![]]);
    let (passages, queries) = embed_task(&mut engine, &task).unwrap();
    assert_eq!(passages.rows, 2);
    assert_eq!(queries.rows, 0);
    assert!(engine.query_calls[0].is_empty());
}
