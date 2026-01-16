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

Language models learn by predicting the next symbol in a sequence. Just like  alinear regressor predicts the value y on a line fromn an input x, the LLM predicts a symbol y from receiving as input a symbol x. So the goal of the LLM is to learn sequences of symbols. In many cases, the symbols are letters in a natural language sentence (letters, or entire words, and this is how the LLM knows that in the sequence 'Apollo 11 landed on the ...', the next word is likely to be 'moon', because if your training set contains multiple books on any topic, the most common next word (highest probability) will be 'moon'.

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

## 3. Tokenization (Character‑Level)

LLMs do not operate on full words, they operate on tokens. Tokens are numbers (usually in fact, vectors of numbers) that are numerical representation of a piece of text. We use numbers because we can run mathematical operations on them. In most cases, a single token is a group of a few letters, but in simplified cases and in complex languages, the token may be down to a single character. Let's adopt this idea and decide that here, each character is a token. 

So our next step is tokenize the text we have, that is convert each unit of text (each character) into a vector (in this case, a single number). A first step is of course to parse the text, and find individual tokens. With more complex tokenization techniques, we first find groups of letters to group together (based on multiple types of criteria). Here, we simply find the letters in our vocabulary, and we just list them:

```

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

encoded = [stoi[c] for c in text]

print("Vocabulary:", chars)
print("Vocabulary size:", vocab_size)
```

You see that our vocabulary is small. It may be interesting to see the tokenization result, i.e., see the token value of each unit of our vocabulary. In practice, we do not care about that value (and a standard LLM may end up with 40,000 tokens, so this would be too much to print), but for curiosity sake, we cane check what tokens we obtained:

```
print("\nCharacter → Token ID mapping:")
for ch in chars:
    print(f"'{ch}' -> {stoi[ch]}")
```

You can see that each character now has a unique ID (its number value). Now our text units (letters) have been converted to tokens (vectors/numbers). This type of transformation is somewhat similar to other AIML techniques where you take some input (e.g., some sensor value) and convert them to some numbers that the AIML process can deal with (for example quantization, or Mel spectrogram transformation that then converts pixels to numbers).

## 4. Training Data Construction

The task of a language model is nothing more than:

    Given the current token, predict the next token.

This prediction can be used in manyw ways, from chatbot dialog to summarization or translation. These various goals are enabled by post training steps, or by the dataset used to train the model. But this general task is identical to time series predictions, or sequence modeling in sensor data (given the current value, predict another value).

So now that we have a vocabulary (in a numeric form that the machine can process), the next step is to train the model, which in essence is still feeding the model with input and expected output pairs, so the model learns the numerical relationship between a given input value and the expected output value.

In a real, large language model, you would feed entire sentences (okay, "groups of tokens") into the model, and in essence (working in fact at token levels) the model would learn, from "Apollo 11 landed on the moon", that if we send "Apollo landed on the ...", the most likely continuation is "moon", and if we say "... landed on the moon", the most likely missing word is "Apollo". Feed millions of sentences, and the model learns probabilities between a given sequence of words and the likely preceding or continuing sequence.

So how do we train a model this way, in practice? Using our toy example, the training is simply about sending to our model a token (a character), and telling it "if I give you this token (this value), then the right answer (the token you need to output) is that token". We take each string in our vocabulary, and we inject it with this logic into the model. For example, we encoded this string as follows:

S T A T E = I D L E \n
↓ ↓ ↓ ↓ ↓   ↓ ↓ ↓ ↓
12 13 2 13 5 1 7 4 8 5 0


In order to train the model, we send the string, character by character (starting from the first one, till the second-to-last one, as there is no character to predict, just a 'stop' decision, if we send the last one). Then for each character we send (for example the first 'S'), we return to the model the right response, which is the next character in the string (in this example, 'T'). Then we send the next character ('T') and tell the model the right response ('A'), etc. Of course, we work with tokens, not the chracters directly, so we send "if 12 -> 13", then "if 13 -> 2", then "if 2 -> 13" etc.

So here, we for the list of inputs (from the first character to the second-to-last one), and the expected output (from the second character, i.e. the correct response if we send the first character), to the last character (the correct response if we send the second-to-last chracter).

We repeat that operation for all strings in our vocabulary:

```

X = encoded[:-1]
Y = encoded[1:]

print("First 10 input tokens:", X[:10])
print("First 10 target tokens:", Y[:10])
```

Now that we have our list of inputs (X) and expected outputs (Y), we can move to the training phase.

## 5. The Model: A Tiny Language Model

The training phase occurs with a classical neural network, where the input is X and the correct output Y. The model then learns the relationships between Y and X. As both are tokens (numeric representation of our chracters), the system can use the techniques that you now understand well to simply find the correct weights/coefficients between each element of X we inject and the correct element of Y.

In more details, this neural network:

    Takes one token as input
    Outputs probabilities for the next token
    Compares its output (with current weight) to the expected output (matching Y element)
    Uses back-ropagation to adjust the weight.
    Uses a hidden layer, just like your keyword‑spotting NN

Important:
This is a language model. There is no transformer, attention, or magic. We could build a more complicated structure, but its principles would eb the same. The more complicated structure may have smarter ways of constructing the relationship between X and Y (with transformers, for example, using multiple neural networks in parallel, with multi-dimensional vector representations of each token, which would allow the set of neural networks to not only learn the relationship in IDLE between I and D, D and L etc., but also between L and I, E and D, E and I etc.) The more complicated structure would be more efficient, but with the same general principles.

Observe the structure of the neural network and note:

    Embedding layer
    Fully connected layers
    Softmax output

```

hidden_size = 64

class TinyCharLM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = TinyCharLM(vocab_size, hidden_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

model
```


## 6. Training the Language Model

We have defined the model, let's now train it to minimize cross‑entropy loss, the same loss used in:

    Classification
    Keyword spotting

Observe the model learning character‑to‑character transitions:

    Loss decreases steadily
    Training is fast because the model is tiny (although we have 400 epochs)

```

epochs = 400

for epoch in range(epochs):
    total_loss = 0.0
    for x, y in zip(X, Y):
        x = torch.tensor([x])
        y = torch.tensor([y])

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {total_loss:.4f}")
```


## 7. Text Generation (The Generative Moment)

Our model is trained, it can predict next tokens. The way it works is the same as other LLMs. Generation works by:

    Feeding a starting token
    Sampling the next token from the output probabilities
    Repeating

This is exactly how large language models generate text. In LLMs, there is often a sort of interface to the user, where the user injects the first tokens (the question you ask to your chatbot). Here, we do not want to design a chatbot interface, just see the principles. Also, our vocabulary is minuscule, so it is highly biased. Some characters have only one solution ('Z' only appears in RECOGNIZED, so if you feed Z, the only possible character is 'E'). Other characters appears about everywhere, and therefore many chracters can be 'the next one', with the same probability (for example, 'E' is in STATE=, RECOGNIZED, IDLE, STATE... if you inject 'E', there are multiple possibilities, including 'stop there' ('EOS', End of Sequence). So be tolerant if the predictor goes a bit all over the place.  

However, observe:

    Output resembles device logs
    Model reproduces learned structure


We inject a character ('S'), then ask the model to generate the next ones (each end-of-sequence results in next line), for 200 characters:

```

def generate_text(start_char, length=200):
    idx = stoi[start_char]
    output = start_char

    for _ in range(length):
        x = torch.tensor([idx])
        logits = model(x)
        probs = torch.softmax(logits, dim=1).detach().numpy()[0]
        idx = np.random.choice(len(probs), p=probs)
        output += itos[idx]

    return output

print(generate_text('S'))
```

Feel free to insert, instead of S, any other letter in our vocabulary, and see the output differences.

## 8. Playing with temperature

In LLMs (and generative models in general), temperature controls the randomness in generation. You apply a "temperature" exponent to the last layers of the model and, in essence, the model predicts the next best token very stricly (low temperature -> return the token with the highest probability), or more losely (higher temperature -> return any token which probability is higher than {whatever the temperature coefficient results into}, even if it is not the 'highst probability' token). 

Temperature is useful, especially for LLMs that are supposed to dialog with the user, because it allows the model to form slightly different responses, even if the same question is asked twice. Temperature provides this impression that the model "thinks" because its answers are not mechanical. In models that needs to provide deterministic (even if 'boring') answers, temperature is set low, in models that need to be creative, temperature is higher. In short:

    Low temperature → predictable
    High temperature → creative but unstable

In our Apollo example, a model with low temperature, when receiving "Apollo", would always respond "11", because this is the most well-known Apollo mission. A model with a higher temperature would compute that possible tokens with reasonably high probability could be "11" (highest probability), "13" (the mission that had an issue withe the famous "Houston we have a problem" - good probability, but lower than 11), or "1" (the first Apollo experiment, famous because a fire killed all the astronauts in the module - good probability). The model would then randmly pick one of these 'good-probability tokens' and return it.

Temperature is useful, but too much of it is bad:

    Too much randomness breaks structure
    Too little randomness causes repetition

See how this principle applies to our toy model:

```
def generate_with_temperature(start_char, length=200, temperature=1.0):
    idx = stoi[start_char]
    output = start_char

    for _ in range(length):
        x = torch.tensor([idx])
        logits = model(x) / temperature
        probs = torch.softmax(logits, dim=1).detach().numpy()[0]
        idx = np.random.choice(len(probs), p=probs)
        output += itos[idx]

    return output

print("Low temperature:")
print(generate_with_temperature('S', temperature=0.5))

print("\nHigh temperature:")
print(generate_with_temperature('S', temperature=1.5))
```

Try to play with the temperature values (in practice, most chatbot temperature is between 1 and 2, anything beyond 2 starts being silly, anything above 3 usually renders the model unusable).

## 9. Embedded Reality Check: Memory & Compute

Now we answer the key question: could we run this on a microcontroller like the XG24? Let's look at this model size:

```
params = sum(p.numel() for p in model.parameters())
memory_kb = params * 4 / 1024

print("Total parameters:", params)
print(f"Model size (float32): {memory_kb:.2f} KB")
```

Less than 10 KB! It could easily fit on our board. But don't get fooled. This is just a toy model. Our current model:
    - Sees **one character at a time**
    - Has **no memory of previous characters**
    - Has **no attention**
    - Has **no notion of context**

Real language models differ in three key ways:

    1. **Context window** (many tokens at once)
    2. **Embedding dimension** (hundreds or thousands)
    3. **Attention layers** (quadratic memory and compute)

let's estimate what happens when we scale toward a *real* LLM. A first step is to scale up our neural network to some more realistic numbers:

```
def estimate_mlp_lm_params(vocab_size, embed_dim, hidden_dim, layers):
    # Embedding
    params = vocab_size * embed_dim
    
    # Hidden layers
    params += embed_dim * hidden_dim
    params += (layers - 1) * hidden_dim * hidden_dim
    params += hidden_dim * vocab_size
    
    return params

vocab_size = 256          # realistic char / byte-level vocab
embed_dim = 256
hidden_dim = 512
layers = 4

params = estimate_mlp_lm_params(vocab_size, embed_dim, hidden_dim, layers)
memory_mb = params * 4 / (1024 ** 2)

print("Scaled MLP Language Model")
print("Total parameters:", params)
print(f"Model size (float32): {memory_mb:.2f} MB")
```

The amount of RAM you get on a constrained board is in the order of 256-512 KB. This 'more realistic' neural network structure is already 4 to 8 times larger than the total amount of RAM you can get on a board.... and that's just a single neural network. In a real network, there are multiple neural networks working in parallel, to learn not only the next/previous character relationships like in our example, but also the farther characters relationships (L with I in IDLE, as we saw above), so that the model can process entire words or sentences. In practice, this parallel structure is called Attention. Attention allows the model not only to learn distant relationship, but also to look at **all tokens in the context window** (the string you inject into the model), thus considering all relationshipsat the same time, and forming an efficient guess on how to continue the dialog.

This gives LLMs:
- Memory
- Reasoning
- Long-range dependencies

But attention scales as:

O(sequence_length² × embedding_dim)

Let's estimate the memory required just for attention activation on our 'more realistic' model. The size of the model, for the inference part, depends very much on how many tokens you want the model to consider simultaneously. In our toy example, we inject one token at a time to illustrate the principle, but in real deployment this would be unusable. It would be as if you injected "A", hoping that the model would guess "Apollo landed on the moon". Just like in a chatbot, you need to be able to inject "at least a few words" for some context to appear. Even in our limited-vocabulary example, you would want to inject "STATE=" to get the most likely state, or even "RECOG..." to get RECOGNIZED=YES.

```
def attention_activation_memory(seq_len, embed_dim, bytes_per_value=4):
    # Q, K, V matrices
    qkv = 3 * seq_len * embed_dim
    
    # Attention matrix (seq_len x seq_len)
    attention = seq_len * seq_len
    
    total_values = qkv + attention
    memory_bytes = total_values * bytes_per_value
    return memory_bytes / (1024 ** 2)

for seq_len in [32, 64, 128, 256]:
    mem = attention_activation_memory(seq_len, embed_dim=256)
    print(f"Sequence length {seq_len:3d}: Attention activations ≈ {mem:.2f} MB")
```

You can see that as soon as you want to allow more than 100 characters for input, the model size explodes beyond 5 MB. This memory is required just to run one forward pass, not counting weights, stack, heap, or firmware. This is why no one runs LLMs on microcontrollers today (until new optimization techniques are invented to squeeze the model size, at least in memory). By contrast, our keyword-spotting neural network was much smaller, because its output was targeted (recognize a few words) and deterministic (classify the audio to 3 or 4 categories)


