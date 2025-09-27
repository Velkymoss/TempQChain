# TempQChain



Fork from https://github.com/HLR/SpaRTUNQChain

Modified code from paper https://arxiv.org/abs/2406.13828 published by Premsri and Kordjamshidi used for temporal relationships between events.



## Setup

### Prerequisites
- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/Velkymoss/SpaRTUNQChain.git
cd SpaRTUNQChain

uv sync
```

### Data
We are using a dense version of the [TimeBank corpus](https://aclanthology.org/P14-2082/). 

Create a data folder in the root of the porject with following structure:
- [TimebankDense.full.txt](https://www.usna.edu/Users/cs/nchamber/caevo/TimebankDense.full.txt)
- timebank_1_2 (folder with annotated TB articles, not freely available unfornatunately)

Then run the create-data script to create the train/dev/test-split for TB-dense:
```bash
q-chain create-data
```


## CLI Usage

The CLI supports FR mode for open-ended questions and YN mode for Yes-No questions about temporal relationships.

## FR Mode
```bash
q-chain temporal-fr [OPTIONS]
```

### YN Mode
```bash
q-chain temporal-yn [OPTIONS]
```

## Command Options

### Training Parameters
- `--epoch INT`: Number of training epochs
- `--lr FLOAT`: Learning rate
- `--batch-size INT`: Batch size for training
- `--check-epoch INT`: Check evaluation every N epochs

### Data Parameters
- `--data-path PATH`: Path to the data folder (default: "data/")
- `--results-path PATH`: Path to save models and predictions (default: "models/")
- `--train-file TEXT`: Training file option: Temp, Origin, SpaRTUN or Human (default: "TEMP")
- `--test-file TEXT`: Test file option: Temp, Origin, SpaRTUN or Human (default: "TEMP")
- `--train-size INT`: Training dataset size
- `--test-size INT`: Test dataset size

### Model Parameters
- `--model TEXT`: Model type to use (bert, roberta, t5-adapter)
- `--dropout`: Enable dropout
- `--constraints`: Enable constraints

### Training Methods
- `--pmd`: Use Primal Dual method
- `--beta FLOAT`: Beta parameter for PMD (default: 0.5)
- `--sampling`: Use sampling loss

- `--sampling-size INT`: Sampling size (default: 1)

### Additional Options
- `--use-chains`: Use chains for data augmentation
- `--text-rules`: Include rules as text
- `--cuda INT`: CUDA device number (-1 for CPU, default: 0)

### Model Loading/Saving
- `--loaded`: Load and evaluate existing model
- `--loaded-file TEXT`: File name to load model from
- `--save`: Save the trained model
- `--save-file TEXT`: File name to save model

## Tests

```bash
uv run pytest
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

**Why this is needed:** The library uses `.index('.')` which throws a `ValueError` when no dot is found in the program name. This happens when using CLI entry points like `q-chain temporal-fr` where the name doesn't have a file extension.