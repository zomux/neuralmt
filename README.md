# neuralmt

A Neural Machine Translation framework for training large-scale networks on multiple nodes with multiple GPUs.

# Requirements

- Python 2.7
- deepy >= 0.2
- theano
- numpy

# An example for WMT15 translation task

1) Clone neuralmt

```
git clone https://github.com/zomux/neuralmt
export PYTHONPATH="$PYTHONPATH:/path/to/neuralmt"
```

2) Create a directory for WMT data

```
export WMT_ROOT="/path/to/your_wmt_folder"
mkdir $WMT_ROOT/text
mkdir $WMT_ROOT/models
```

3) Tokenize de-en training corpus, and rename them to following filenames

- $WMT_ROOT/text/wmt15.de-en.de
- $WMT_ROOT/text/wmt15.de-en.en

4) Build training data

```
cd /path/to/neuralmt
python examples/gru_search/preprocess.py
```

5) Train on 3 GPUs

```
python -m deepy.multigpu.launch examples/gru_search/train.py gpu0 gpu1 gpu2
```

6) Wait for several days

7) Test your model

```
python examples/gru_search/test.py
```

(The test script only translate one sample sentence, you can modify it to translate a text file)


# Note

Training on multiple machine is still in development.

Although the current framework for parallelism shall be extended to multiple machine easily, it require some works.

# Some Results

- WMT15 German-English task (using the model in the example)
 - BLEU: 21.29
 - Duration: 2.5 days with 3 Titan X GPUs

Raphael Shu, 2016