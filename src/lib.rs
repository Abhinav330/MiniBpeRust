//! Native backend for the `MiniBpeRust` Python package.
//!
//! The Python layer (`MiniBpeRust.RegexTokenizer`) owns the file format, the
//! special-token handling and decoding. This crate only does the two things
//! that are expensive in pure Python:
//!
//! * `train_on_text` / `train_on_files` - byte-level BPE training that
//!   reproduces minbpe's tie-breaking exactly.
//! * `RustEncoder` - encoding from an ordered merge list, optionally in
//!   parallel across documents.
//!
//! Long running calls release the GIL, so Python threads keep running while
//! Rust trains or encodes.

mod encode;
mod train;

use encode::Encoder;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::Python;

fn to_py_err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

/// Train on a text string. Returns the merges in order: [((id1, id2), count), ...]
#[pyfunction]
#[pyo3(signature = (text, pattern, vocab_size, verbose=false))]
fn train_on_text(
    py: Python<'_>,
    text: &str,
    pattern: &str,
    vocab_size: u32,
    verbose: bool,
) -> PyResult<Vec<((u32, u32), u64)>> {
    py.detach(|| train::train_on_text(text, pattern, vocab_size, verbose))
        .map_err(to_py_err)
}

/// Train on text files read directly by Rust (best for a huge corpus).
#[pyfunction]
#[pyo3(signature = (paths, pattern, vocab_size, verbose=false))]
fn train_on_files(
    py: Python<'_>,
    paths: Vec<String>,
    pattern: &str,
    vocab_size: u32,
    verbose: bool,
) -> PyResult<Vec<((u32, u32), u64)>> {
    py.detach(|| train::train_on_files(&paths, pattern, vocab_size, verbose))
        .map_err(to_py_err)
}

/// Fast encoder built from a split pattern and an ordered list of merges.
#[pyclass(module = "MiniBpeRust.rustminibpe")]
struct RustEncoder {
    inner: Encoder,
}

#[pymethods]
impl RustEncoder {
    /// `merges` must be in merge order: index 0 is the first merge (id 256).
    #[new]
    fn new(pattern: &str, merges: Vec<(u32, u32)>) -> PyResult<Self> {
        let inner = Encoder::new(pattern, &merges).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Number of merges this encoder was built with.
    #[getter]
    fn num_merges(&self) -> usize {
        self.inner.num_merges()
    }

    /// Encode one string, ignoring special tokens.
    fn encode_ordinary(&self, py: Python<'_>, text: &str) -> PyResult<Vec<u32>> {
        py.detach(|| self.inner.encode_ordinary(text))
            .map_err(to_py_err)
    }

    /// Encode many documents in parallel on all cores.
    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        py.detach(|| self.inner.encode_batch(&texts))
            .map_err(to_py_err)
    }

    fn __len__(&self) -> usize {
        self.inner.num_merges()
    }

    fn __repr__(&self) -> String {
        format!("RustEncoder(num_merges={})", self.inner.num_merges())
    }
}

#[pymodule]
fn rustminibpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(train_on_text, m)?)?;
    m.add_function(wrap_pyfunction!(train_on_files, m)?)?;
    m.add_class::<RustEncoder>()?;
    Ok(())
}
