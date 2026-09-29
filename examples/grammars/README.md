# Grammar corpus fixtures

`branching-v1` is the smallest stochastic source used to verify the OpenGrm
corpus path. Its two successful paths have equal probability. Compile the text
FST with `fstcompile`, preserving its symbol tables, before passing the binary
model to `./ennx corpus grammar`:

```text
./ennx corpus grammar model.fst --out corpus/run-1 --seed 7 --sequences 100000
```

The command runs from either ENNX build route. Buck2/Reindeer builds the Rust
orchestrator; the orchestrator asks Bazel for the pinned OpenGrm and OpenFst
C++ binaries. `samples.txt` contains one output-label sequence per line and
`manifest.json` records the input and output digests, revisions, and sampling
contract.
