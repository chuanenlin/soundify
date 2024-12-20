# Soundify

Soundify matches audio clips to video. The tool builds on [CLIP](https://openai.com/blog/clip/) to classify scenes (e.g., "bicycle") and retrieves corresponding audio files (e.g., bicycle.wav). Users may layer sounds to create depth and add an additional ambient sound. Soundify is context-aware, being able to adapt retrieved audio files with appropriate panning and volume in a fine-grained manner. The prototype uses [Streamlit](https://streamlit.io/) for its UI.

---

## Setting up 

1.  Clone the repository.

```python
git clone https://github.com/runwayml/soundify.git
cd soundify
```

2.  Install package dependencies.

```python
pip install -r requirements.txt
```

3.  Download the [sound samples](https://drive.google.com/file/d/1Ag1bcTJgJIDn92afHja86zxGt_YDgUta/view?usp=sharing), unzip, and save them under the **sound** directory.

    You may add your own sound samples (in .wav format) by adding them under the **sound** directory and updating **main-sounds.txt** or **ambient-sounds.txt** with their filenames (without filename extension).

1.  Download the [demo video](https://drive.google.com/file/d/1zaqumFFkAavdAwO-pkgn_xUPiRgpz5iA/view?usp=sharing) and save it under the root directory.

---

## Running

```python
streamlit run soundify.py
```

---

## Updates

TBD
