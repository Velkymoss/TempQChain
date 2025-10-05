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

## TODO
- create domiknows programs for T5-v4-FR and T5-v5-FR
- create tests for graphs