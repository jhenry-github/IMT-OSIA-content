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

