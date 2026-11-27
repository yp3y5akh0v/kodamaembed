//! The documents' model tables list the registered models in order, with the registry's facts.

use kodamaembed::{EmbeddingModel, RerankerModel};

const DOCUMENTS: [(&str, &str); 3] = [
    ("README.md", include_str!("../../../../README.md")),
    (
        "docs/integration.md",
        include_str!("../../../../docs/integration.md"),
    ),
    (
        "python/README.md",
        include_str!("../../../../python/README.md"),
    ),
];

/// What a model table may say about one registered model.
struct Facts {
    repository: &'static str,
    api_name: String,
    kind: &'static str,
    dimensions: String,
    tokens: usize,
    download: u64,
    revision: &'static str,
}

fn registry() -> Vec<Facts> {
    let embedders = EmbeddingModel::ALL.iter().map(|&model| Facts {
        repository: model.repository(),
        api_name: format!("EmbeddingModel::{model:?}"),
        kind: "embedding",
        dimensions: model.dim().to_string(),
        tokens: model.spec().max_length,
        download: model.download_size(),
        revision: model.revision(),
    });
    let rerankers = RerankerModel::ALL.iter().map(|&model| Facts {
        repository: model.repository(),
        api_name: format!("RerankerModel::{model:?}"),
        kind: "reranker",
        dimensions: "-".into(),
        tokens: model.max_length(),
        download: model.download_size(),
        revision: model.revision(),
    });
    embedders.chain(rerankers).collect()
}

/// As `kodamaembed models` prints sizes: decimal units, one decimal.
fn size(bytes: u64) -> String {
    if bytes < 1000 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64 / 1000.0;
    for unit in ["kB", "MB"] {
        if value < 999.95 {
            return format!("{value:.1} {unit}");
        }
        value /= 1000.0;
    }
    format!("{value:.1} GB")
}

fn expected(column: &str, facts: &Facts) -> String {
    match column {
        "Model" | "Name" => format!("`{}`", facts.repository),
        "Name in the API" => format!("`{}`", facts.api_name),
        "Kind" => facts.kind.into(),
        "Dimensions" => facts.dimensions.clone(),
        "Tokens" => facts.tokens.to_string(),
        "Download" => size(facts.download),
        "Pinned commit" => format!("`{}`", facts.revision),
        other => panic!("no registry fact for the column {other}"),
    }
}

/// Each Markdown table as rows of trimmed cells, its header first.
fn tables(document: &str) -> Vec<Vec<Vec<&str>>> {
    let mut tables = Vec::new();
    let mut table = Vec::new();
    for line in document.lines().chain([""]) {
        match line
            .trim()
            .strip_prefix('|')
            .and_then(|row| row.strip_suffix('|'))
        {
            Some(row) => table.push(row.split('|').map(str::trim).collect()),
            None if !table.is_empty() => tables.push(std::mem::take(&mut table)),
            None => {}
        }
    }
    tables
}

#[test]
fn each_model_table_lists_the_registry_in_order_with_its_facts() {
    let registry = registry();
    let names: Vec<String> = registry
        .iter()
        .map(|facts| format!("`{}`", facts.repository))
        .collect();
    for (document, text) in DOCUMENTS {
        let mut models = tables(text).into_iter().filter(|table| {
            table
                .get(2)
                .is_some_and(|row| names.contains(&row[0].into()))
        });
        let table = models
            .next()
            .unwrap_or_else(|| panic!("{document} has no model table"));
        assert!(models.next().is_none(), "{document} has two model tables");
        let (header, rows) = (&table[0], &table[2..]);
        assert_eq!(rows.len(), registry.len(), "{document}");
        for (row, facts) in rows.iter().zip(&registry) {
            assert_eq!(row.len(), header.len(), "{document}: {row:?}");
            for (column, cell) in header.iter().zip(row) {
                assert_eq!(
                    *cell,
                    expected(column, facts),
                    "{document}: {column} of {}",
                    facts.repository
                );
            }
        }
    }
}

#[test]
fn sizes_print_as_the_command_line_prints_them() {
    assert_eq!(size(612), "612 B");
    assert_eq!(size(466_247), "466.2 kB");
    assert_eq!(size(91_335_235), "91.3 MB");
    assert_eq!(size(999_960_000), "1.0 GB");
    assert_eq!(size(1_341_083_308), "1.3 GB");
}
