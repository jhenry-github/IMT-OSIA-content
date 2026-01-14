# Lab setup
This lab uses Pytorch, because it is the common framework for LLM work. Other labs in this course have been using Tensorflow, and both frameworks suppose different library dependencies. A first step to run this lab is therefore to create a new environment. If you are still in the Tensorflow environment, you first need to exit to return to the native Linux CLI prompt. Make sure to close your Jupyter notebook first, then if the prompt has the (IMTAIML) marker, deactivate the Tensorflow environment to exit back to standard CLI:

```shell
(IMTAIML) developer@lmvm-xfce:~/IMTAIML$ deactivate
developer@lmvm-xfce:~/IMTAIML$
```

Go up one level, and create a new environment for Pytorch:

```shell
developer@lmvm-xfce:~/IMTAIML$ cd ..
developer@lmvm-xfce:~$python3 -m venv IMTPytorch
```

The operation should take a few seconds, because venv creates a lot of default elements in the environement, in addition to the environement itself. Once the operation completes, you should see a new IMTPytorch folder. Change to this directory:

```shell
cd IMTPytorch
```

Once there, you can activate your environment. This activation is done by calling the activate operation with the source command:

```shell
source bin/activate
```

You should see that your prompt has changed:

```shell
(IMTPytorch) developer@lmvm-xfce:~/IMTPytorch$
```

The parenthesis show in which environement context you are operating. We will install a few tools, and will install other tools later from Jupyter. For now, install Pytorch, Jupyter (yes, you need to install Jupyter again, as each environment is self-contained): 

```shell
pip install torch jupyter
```

Accept the packages proposed by the installer. A lot of dependencies need to be installed, so this will take a while. It may be that some packages will end up being required, you can add them from within Jupyter later.


Once the installation completes, launch Jupyter:

```shell
jupyter-lab &
```

# From Embedded ML to Generative AI

## Big Picture

So far in this course, you have trained:

    Regression models → predict numbers
    Classifiers → predict labels (yes / no)
    Neural networks → learn nonlinear decision boundaries

In this lab, you will see that language models are built using exactly the same principles.

The only difference is scale.

By the end of this lab, you will:

    Train a generative model that produces text
    Understand what an LLM actually does
    Quantitatively prove why LLMs do not fit on microcontrollers
    Learn how LLMs are realistically used with embedded systems


## 1. Environment Setup

This cell imports the same libraries you have already used in previous neural‑network labs. Although these imports are necessary, onf of the goals of this cell is to confirm that nothing “special” is required to build a language model. An LLM is 'just' a bunch of neural networks organized in a specific way to predict a specific type of output. You should observe that we use standard standard numerical and deep‑learning libraries, nothing more:

```

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
np.random.seed(42)
```

As in many of the previous labs, we fix the 'randomness' of the random numbers we will generate with a seed, so you see the same results that we see in class. In the real world, you would want 'random' to be really random, so you would not use those seed commands.


## 2. Dataset: Embedded‑Style Device Logs

Language models learn by predicting the next symbol in a sequence. Just like  alinear regressor predicts the value y on a line fromn an input x, the LLM predicts a symbol y from receiving as input a symbol x. So the goal of the LLM is to learn sequences of symbols. In many cases, the symbols are letters in a natural language sentence (letters, or entire words, and this is how the LLM knows that in the sequence 'Appollo 11 landed on the ...', the next word is likely to be 'moon', because if your training set contains multiple books on any topic, the most common next word (highest probability) will be 'moon'.

in this lab, we could use English sentence, but as we are in the world of embedded systems, our words could instead be IoT-related messages. So instead of natural language, we will use embedded device logs, which:

    Are structured
    Are repetitive
    Look like real firmware output

This makes the behavior of the model easier to interpret in our context. So let's first create a minimal text dataset that resembles embedded system output. This is shorter than what a normal LLM will be trained on (a standard LLM is trained on billions of pages of text extracted from multiple sources), but the principles are the same: in the end, there are sequences of words, and the model learns how these words connect to one another.

```

text = """
STATE=IDLE
STATE=LISTEN
STATE=LISTEN
STATE=RECOGNIZED_YES
STATE=IDLE
STATE=LISTEN
STATE=RECOGNIZED_NO
STATE=IDLE
""".strip()

print(text)
```

You can observe that our vacabulary is very small, and yet the pattern is very repetitive. Not only some words tend to come back often (just like in the English language), but also the sequence of letters is limited (something that starts with 'I' is likely to be 'IDLE'). In the English language, there is more variety of course, but the same principle appplies (and the LLMs operate at syllable level): if your sequence starts with moo..., there is a high probability that the word will be 'moon'.

What to observe:

    Very small vocabulary
    Repetitive patterns

