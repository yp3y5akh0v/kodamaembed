# kodamaembed-model

Shared encoder shapes, tensor validation, output modes and profiling types
for KodamaEmbed backends. This crate also provides the double-precision
reference forward implementation used by kernel tests.

The `repro` module defines the plain-data export of a calibrated centered
INT8 encoder. It carries the weights and tables another backend needs to
implement the same numerical contract.

[`kodamaembed`](https://crates.io/crates/kodamaembed) loads checkpoints and
runs inference;
[architecture](https://github.com/yp3y5akh0v/kodamaembed/blob/master/docs/architecture.md)
describes crate boundaries and numerical identity.

```text
cargo test -p kodamaembed-model --locked
cargo doc -p kodamaembed-model --no-deps --open
```

[Apache-2.0](https://github.com/yp3y5akh0v/kodamaembed/blob/master/LICENSE).
