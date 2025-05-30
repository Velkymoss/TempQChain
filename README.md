# SPARTUNQChain



Fork from https://github.com/HLR/SpaRTUNQChain

Refactored code from paper https://arxiv.org/abs/2406.13828 puplished by Premsri and Kordjamshidi

## Dependencies



```bash
make install
```

# Experiment


The program will save the parameters of the model in Models folder for any further use.

Data available here: https://drive.google.com/drive/folders/16nBxg1xcPfuQu58Df-PSQZYABsgmk9KQ?usp=sharing.
Note that the augmented Q-Chain part in train_YN_v3.json and train_FR_v3.json on fact_infos parameters


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