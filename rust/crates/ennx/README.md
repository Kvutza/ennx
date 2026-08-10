# ENNX

Epistemic nearest-neighbor Bayesian optimization.

```sh
cargo add ennx
```

The default build includes Faiss, BPANN, Metal on macOS, and OpenCL. Install
Faiss first or set `FAISS_LIB_DIR`. It also includes the Tracy client; connect a
Tracy 0.13.1 profiler to any process using ENNX.
