# SPARTUNQChain



Fork from https://github.com/HLR/SpaRTUNQChain

Refactored code from paper https://arxiv.org/abs/2406.13828 puplished by Premsri and Kordjamshidi used for temporal events.



## Dependencies



```bash
make install
```

# Experiment


The program will save the parameters of the model in Models folder for any further use.

For the data see data/README.md


### Supported models:
- `bert` (default)
- `roberta`
- `t5-adapter`

### Example usage:

```bash
make primal-dual-yn MODEL=roberta
```

If `MODEL` is not specified, `bert` will be used by default.

---

## Yes-No Question Experiments

### Baseline
```bash
make baseline-yn MODEL=<model_name>
```

### Primal-Dual
```bash
make primal-dual-yn MODEL=<model_name>
```

### Primal-Dual + Q-Chain
```bash
make primal-dual-qchain-yn MODEL=<model_name>
```

## Experiment with FR

The possible model option is [ "bert"].

### Baseline
```bash
make baseline-fr
```

### Primal-Dual
```bash
make primal-dual-fr
```

### Primal-Dual + Q-Chain
```bash
make primal-dual-qchain-fr
```

## Tests

```bash
make tests
```
## Known Issues

### DomiKnows Library Bug Fix

Due to a bug in the `domiknows==0.533` library, you need to manually fix one line of code after installation:

**File:** `.venv/lib/python3.12/site-packages/domiknows/program/program.py` (line ~37)

**Change from:**
```python
if self.programName.index('.') >= 0:
```

**Change to:**
```python
if '.' in self.programName:
```

**Why this is needed:** The library uses `.index('.')` which throws a `ValueError` when no dot is found in the program name. This happens when using CLI entry points like `q-chain temporal-fr` where the name file doesn't have a file extension.