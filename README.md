# CS6320 N-gram Language Model

This repository contains the implementation of an **N-gram Language Model** built as part of the **CS6320: Natural Language Processing** group project at UT Dallas. The project was completed by a team of four students.

The model supports:

- **Unigram and Bigram models** (extensible to higher n-grams).
- **Smoothing techniques**: Laplace (add-1), Add-k, or none.
- **Unknown word handling** via frequency thresholding or regex-based replacement.
- **Perplexity evaluation** on validation data.
- **Configurable hyperparameters** for experimentation.

---

## Features

- Preprocess text into tokens (lowercasing, punctuation stripping).
- Handle unknown tokens using either:
  - Frequency thresholding.
  - Regex rules (e.g., numbers, very short/long words).
- Train an N-gram language model with context counts.
- Compute probabilities with different smoothing methods.
- Evaluate model quality via **perplexity**.
- Example probability calculations for sample n-grams.

---

## File Structure

```

├── n_gram.py # Main implementation of the N-gram language model
├── README.md # Project documentation (this file)
├── DATASET/ # Training and validation corpus files (not included here)
└── REPORT.pdf # Final project report

```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Kartik11082/CS6320-Group-14-A-1.git
cd CS6320-Group-14-A-1
```

Ensure you have **Python 3.8+** installed. Install any required dependencies:

---

## Usage

Run the model from the command line:

```bash
python n_gram.py --train data/train.txt --validation data/valid.txt --n 2 --smoothing laplace
```

### Arguments

- `--train` : Path to the training corpus (required).
- `--validation` : Path to the validation corpus (required).
- `--n` : Order of n-gram model (`1` for unigram, `2` for bigram, etc.). Default = 2.
- `--smoothing` : Smoothing method (`laplace`, `add_k`, `none`). Default = `laplace`.
- `--k` : Add-k smoothing parameter. Default = 1.0.
- `--unk_threshold` : Frequency threshold for replacing tokens with `<UNK>`. Default = 1.
- `--unk_method` : Unknown word handling (`threshold`, `regex`). Default = `threshold`.

---

## Example

Training a bigram model with Laplace smoothing:

```bash
python n_gram.py --train DATASET\train.txt --validation DATASET\val.txt --n 2 --smoothing none
```

Example output:

```
TTraining 2-gram model...
First pass: counting word frequencies...
Vocabulary size: 6323
Second pass: collecting n-gram counts...
Total n-grams collected: 38377
Total contexts collected: 6321
Calculating perplexity...

Perplexity on validation set: 33172.01

Example probability calculations:
P(hotel|the) = 0.078028
P(was|hotel) = 0.055126
P(good|was) = 0.007667
P(the|<s>) = 0.093750
P(</s>|good) = 0.000000
```

---

## Project Report

A detailed **report (report.pdf)** is included, which covers:

- Implementation details.
- Experiments with unigram/bigram models.
- Effect of smoothing methods on perplexity.
- Error analysis and limitations.

---

## Authors

This project was developed as part of **CS6320 (Natural Language Processing)** at UT Dallas.
Group Members:

1. Kartik Karkera
2. David Song
3. Dhruval patel
4. Rahul Patil
