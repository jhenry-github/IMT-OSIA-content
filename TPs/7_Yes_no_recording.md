# Audio Recording for Yes-No keyword spotting

This notebook comes in support of the keyword spotting lab, where we train our own tfLite model to recognize a keyword like yes and no. Most of the training is done in Google Colab, but this notebook helps you record your own yes and no samples. 
First, let's make sure you have the right packages installed:

```shell
!pip install sounddevice soundfile numpy
```

We installed these packages because we want to use them. So let's call them in:

```shell
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import zipfile
```

Next, we define a simple function that collects sound snippets fromn your computer microphone, by chunks of one second centered at 16,000 Hz, and saves each snippet in a wav file:

```shell
SAMPLE_RATE = 16000
DURATION = 1.0  # seconds

def record_sample(label, index):
    print(f"Recording '{label}' sample #{index}...")
    audio = sd.rec(
        int(SAMPLE_RATE * DURATION),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    os.makedirs(f"student_data/{label}", exist_ok=True)
    filename = f"student_data/{label}/{label}_{index}.wav"
    sf.write(filename, audio.squeeze(), SAMPLE_RATE)

    print("Saved:", filename)
```

We then just need to call the function by default five times for each keyword, but feel free to change this number if you want:

```shell
NUM_SAMPLES = 5  # per word

for label in ["yes", "no"]:
    for i in range(NUM_SAMPLES):
        input(f"Press Enter to record '{label}' sample {i+1}/{NUM_SAMPLES}")
        record_sample(label, i)
```

To make the result easy to manipulate and in particular to upload to Google Colab, let's collect all these samples and zip them into a single archive:

```shell
zip_path = "student_yes_no_samples.zip"

with zipfile.ZipFile(zip_path, "w") as z:
    for root, _, files in os.walk("student_data"):
        for f in files:
            full_path = os.path.join(root, f)
            z.write(full_path)

print("Created:", zip_path)
```

That's it for this part. The next step occurs in Google colab where we use those sound snippets along with a few more that exist in a database of pre-recorded words to create our model. 



