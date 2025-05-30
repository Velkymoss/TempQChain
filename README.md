# SPARTUNQChain



Fork from https://github.com/HLR/SpaRTUNQChain

Refactored code from paper https://arxiv.org/abs/2406.13828 puplished by Premsri and Kordjamshidi

## Dependencies



```bash
make install
```

# Experiment

The possible model option is ["roberta", "t5-adapter", "bert"].

The program will save the parameters of the model in Models folder for any further use.

Data available here: https://drive.google.com/drive/folders/16nBxg1xcPfuQu58Df-PSQZYABsgmk9KQ?usp=sharing.
Note that the augmented Q-Chain part in train_YN_v3.json and train_FR_v3.json on fact_infos parameters


## Yes-No Question

### Baseline
```bash
make baseline-yn
```
### Primal-Dual
```bash
make primal-dual-yn
```
### Primal-Dual + Q-Chain
```bash
make primal-dual-qchain-yn
```

## Experiment with FR
The possible model option is [ "bert"].
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
