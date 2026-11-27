//! The dependency versions the fingerprint hashes against `Cargo.lock`.

/// A pin bumped without its hashed version would keep the old fingerprints.
#[test]
fn hashed_dependency_versions_are_the_locked_ones() {
    let lock = include_str!("../../../../Cargo.lock");
    let version = |name: &str| {
        let heading = format!("name = \"{name}\"");
        let versions: Vec<&str> = lock
            .lines()
            .zip(lock.lines().skip(1))
            .filter(|(line, _)| *line == heading)
            .map(|(_, next)| {
                next.trim_start_matches("version = \"")
                    .trim_end_matches('"')
            })
            .collect();
        assert_eq!(versions.len(), 1, "{name}: {versions:?}");
        versions[0]
    };
    let hashed = format!(
        "b\"tokenizers={};half={}\\0\"",
        version("tokenizers"),
        version("half")
    );
    assert!(
        include_str!("../../src/lib.rs").contains(&hashed),
        "{hashed}"
    );
}
