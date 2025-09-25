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

## Transitivity Rules Table for TB-Dense

### Rule 1

| Relation 1   | Relation 2   | Resulting Relations |
|--------------|--------------|---------------------|
| before       | before       | before              |
| after        | after        | after               |
| includes     | includes     | includes            |
| is included  | is included  | is included         |
| simultaneous | simultaneous | simultaneous        |
| vague        | vague        | vague               |

### Rule 2 

| Relation 1   | Relation 2   | Resulting Relations |
|--------------|--------------|---------------------|
| before       | simultaneous | before              |
| after        | simultaneous | after               |
| includes     | simultaneous | includes            |
| is included  | simultaneous | is included         |
| vague        | simultaneous | vague               |
| vague        | before       | vague               |
| vague        | after        | vague               |
| vague        | includes     | vague               |
| vague        | is included  | vague               |

### Rules for "before"

| Relation 1   | Relation 2   | Resulting Relations                                      |
|--------------|--------------|----------------------------------------------------------|
| before       | after        | before, after, includes, is included, simultaneous, vague|
| before       | includes     | before, includes, vague                                  |
| before       | is included  | before, is included, vague                               |
| before       | vague        | before, includes, is included, vague                     |
| before       | vague        | vague                                                    |

### Rules for "after"

| Relation 1   | Relation 2   | Resulting Relations                                      |
|--------------|--------------|----------------------------------------------------------|
| after        | before       | before, after, includes, is included, simultaneous, vague|
| after        | includes     | after, includes, vague                                   |
| after        | is included  | after, is included, vague                                |
| after        | vague        | after, includes, is included, vague                      |
| after        | vague        | vague                                                    |

### Rules for "includes"

| Relation 1   | Relation 2   | Resulting Relations                                      |
|--------------|--------------|----------------------------------------------------------|
| includes     | before       | before, includes, vague                                  |
| includes     | after        | after, includes, vague                                   |
| includes     | is included  | before, after, includes, is included, simultaneous, vague|
| includes     | vague        | before, after, includes, vague                           |
| includes     | vague        | vague                                                    |

### Rules for "is included"

| Relation 1   | Relation 2   | Resulting Relations                                      |
|--------------|--------------|----------------------------------------------------------|
| is included  | before       | before, is included, vague                               |
| is included  | after        | after, is included, vague                                |
| is included  | includes     | before, after, includes, is included, simultaneous, vague|
| is included  | vague        | before, after, is included, vague                        |
| is included  | vague        | vague                                                    |

### Rules for "simultaneous"

| Relation 1   | Relation 2   | Resulting Relations |
|--------------|--------------|---------------------|
| simultaneous | before       | before              |
| simultaneous | after        | after               |
| simultaneous | includes     | includes            |
| simultaneous | is included  | is included         |
| simultaneous | vague        | vague               |



## TODOs

- On dev-max branch we already have programs for tb_dense available for YN and FR.
There still need to be implemented programs for the various T5 versions
- Creation of dev and test set for tb-dense
- Because of our non-deterministic transitive rules our YN graph might work differently. 
We need to verify if our current tb-dense YN graph is valid because of this.