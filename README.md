# Tweet Sentiment RNN

A PyTorch-based NLP project for **3-class sentiment classification** of tweets using the [TweetEval Sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval) dataset.

The project implements a **unidirectional vanilla RNN** with an embedding layer. Tweet text is tokenized using the `bert-base-uncased` tokenizer, while the classification model itself is built from PyTorch RNN components.

## Model

* Tokenizer: `BertTokenizerFast` (`bert-base-uncased`)
* Maximum sequence length: 64
* Embedding dimension: 128
* RNN: PyTorch `nn.RNN`
* Hidden dimension: 128
* RNN layers: 1
* Bidirectional: No
* Sequence representation: Mean pooling over RNN outputs
* Number of classes: 3
* Loss: `CrossEntropyLoss`
* Optimizer: SGD
* Learning rate: 0.1
* Momentum: 0.9
* Batch size: 32

### Sentiment Classes

* `0` — Negative
* `1` — Neutral
* `2` — Positive

## Dataset

The project uses the **TweetEval Sentiment** dataset with:

* Training split
* Validation split
* Test split

The text is tokenized and padded/truncated to a maximum length of 64 tokens.

## Training

The model is trained for **10 epochs** and evaluated using multiclass accuracy.

The notebook also plots:

* Training vs. validation loss
* Training vs. validation accuracy

The model with the best validation loss is saved as a PyTorch checkpoint and later reloaded for evaluation on the test set.

## Requirements

```bash
pip install torch transformers datasets torchmetrics tqdm matplotlib numpy
```

## Notebook

The complete implementation is available in:

```text
RNN_NLP_Classification_TweetEval.ipynb
```

The notebook was developed in **Google Colab** using PyTorch.
