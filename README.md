# DistilBERT NER App

A tiny Named Entity Recognition (NER) workflow that fine-tunes DistilBERT on CoNLL-2003 and serves an interactive UI with Streamlit. Entities are rendered with a simple displaCy-style highlight.

## What’s inside
- Training notebook: `main.ipynb` (loads CoNLL-2003, tokenizes, trains with Hugging Face Trainer)
- Inference app: `app.py` (Streamlit UI + model loader from `./results/checkpoint/`)
- Checkpoints (local): `./results/checkpoint-*/`

## Quickstart
1) Create and activate a virtual environment (recommended)
2) Install dependencies
3) Run the app

```powershell
# from repo root
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Notes
- The app expects a trained checkpoint under `./results/checkpoint/` (config.json, model.safetensors, tokenizer files, etc.) which can be found at https://drive.google.com/drive/folders/1ILt4F3R9Y9z4nu_GlTg-V1k6ENqiNWc2?usp=sharing.
- If you’re cloning this repo fresh, place your fine-tuned model in that folder or update `model_path` in `app.py`.

## Training (optional)
Open `main.ipynb` and run the cells. It will:
- Load the dataset (CoNLL-2003 via Hugging Face Datasets)
- Tokenize and align labels
- Train DistilBERT with the Hugging Face Trainer API
- Write checkpoints to `./results/`

Adjust training arguments (batch size, epochs, fp16, etc.) as you like.

## Project structure
```
app.py
main.ipynb
results/
  checkpoint/
static/
docs/
```

## References
- DistilBERT: Sanh et al., DistilBERT, a distilled version of BERT (arXiv:1910.01108)
  - https://arxiv.org/abs/1910.01108
- Hugging Face Transformers
  - https://huggingface.co/docs/transformers
- Hugging Face Datasets
  - https://huggingface.co/docs/datasets
- CoNLL-2003 dataset on HF
  - Canonical: https://huggingface.co/datasets/conll2003
  - Variant used in notebook: https://huggingface.co/datasets/eriktks/conll2003
- SeqEval (sequence labeling metrics)
  - https://github.com/chakki-works/seqeval
- Streamlit
  - https://streamlit.io/
- safetensors
  - https://github.com/huggingface/safetensors

## License

MIT

