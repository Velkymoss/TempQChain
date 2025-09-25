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

## TODOs

- On dev-max branch we already have programs for tb_dense available for YN and FR.
There still need to be implemented programs for the various T5 versions
- Creation of dev and test set for tb-dense
- Because of our non-deterministic transitive rules our YN graph might work differently. 
We need to verify if our current tb-dense YN graph is valid because of this.