# The four parsers of the benchmark

Every parser is run with the recipe from its own paper/repository; they are not forced into a
shared configuration. All of them are scored the same way, with `conll18_ud_eval`
(UPOS / UAS / LAS, LAS ignoring language-specific deprel subtypes), on the same three test
sets (OTA-BOUN, DUDU, TR-BOUN) and the same CV folds (`src/run_sweep.py:make_folds`).

Results of every run are written under `$STEPS_OUTPUT_DIR` (point it at Drive on Colab),
one folder per (parser, encoder, train set, fold, seed), each with `results.json`, the
predictions and the logs. Re-running a command skips finished runs.

| Parser | Runner | Environment | Encoders |
|---|---|---|---|
| MaChAmp 0.4 | `src/run_machamp.py` | transformers 4.x + torch | any HuggingFace model |
| STEPS | `src/run_sweep.py` | transformers 3.1.0 + torch (`.venv`) | BERT / RoBERTa / XLM-R |
| UDPipe 2 | `src/run_udpipe2.py` | TensorFlow 2 + torch + transformers | any HuggingFace model (embeddings service) |
| UDify | `src/run_udify.py` | Python 3.8, torch 1.4, allennlp 0.9 | BERT family only |

## MaChAmp

```bash
pip install "transformers>=4.40,<5" jsonnet sentencepiece
python src/run_machamp.py setup
python src/run_machamp.py run --models berturk --train-sets ota --cv 5
```
Config: `configs/machamp/params.json` (MaChAmp's own defaults, as used for its published UD
results). Patches in `patches/machamp-*.patch`: early stopping, and tokenizer/encoder fixes
needed for ModernBERT models (TabiBERT, mmBERT).

## STEPS

```bash
# its own environment: modern torch (with its CUDA packages) + the old transformers
uv venv -p 3.9 .venv-steps
uv pip install --python .venv-steps/bin/python torch          # NOT with --no-deps: CUDA libs
uv pip install --python .venv-steps/bin/python "transformers==3.1.0" --no-deps
uv pip install --python .venv-steps/bin/python numpy packaging requests regex tqdm filelock \
    sentencepiece sacremoses tokenizers mlflow pyconll networkx sympy
.venv-steps/bin/python src/run_sweep.py run --tasks parse upos --models berturk --train-sets ota --cv 5
```
The BERTurk folder in the repo holds LFS pointers, so fetch the weights and give the tokenizer
its config (without it, transformers 3.1.0 lowercases and strips accents):
```bash
curl -sL -o data/pretrained_embeddings/bert-base-turkish-cased/pytorch_model.bin \
  https://huggingface.co/dbmdz/bert-base-turkish-cased/resolve/main/pytorch_model.bin
echo '{"do_lower_case": false, "max_len": 512}' > data/pretrained_embeddings/bert-base-turkish-cased/tokenizer_config.json
```
Config: `configs/ota_boun.json` (parsing) and `configs/ota_upos.json` (POS), i.e. the settings
of the previous paper. Needs the old stack (transformers 3.1.0); the local `.venv` has it.

## UDPipe 2

```bash
pip install tensorflow tf-keras torch "transformers>=4,<5" ufal.chu_liu_edmonds
python src/run_udpipe2.py setup
python src/run_udpipe2.py run --models berturk --train-sets ota --cv 5
```
Recipe: UDPipe 2's defaults (60 epochs: 40 at lr 1e-3 + 20 at 1e-4, batch 32, BiLSTM tagger
and parser over frozen contextual embeddings = the mean of the encoder's last four layers).
Embeddings are computed once per encoder + corpus and cached in `udpipe2_embeddings/`.
`patches/udpipe2-tf2.patch` makes training run on TensorFlow 2 (plain Adam instead of the
removed `tf.contrib` LazyAdam, no TensorBoard summaries);
`patches/udpipe2-wembeddings-models.patch` registers our encoders in the embeddings service.

## UDify

```bash
bash scripts/setup_udify_env.sh           # Python 3.8 + torch 1.4 + allennlp 0.9
.venv-udify/bin/python src/run_udify.py setup
.venv-udify/bin/python src/run_udify.py run --models berturk --train-sets ota --cv 5
```
Recipe: UDify's `udify_bert_finetune` configuration (multi-task UPOS/FEATS/LEMMAS/DEPS over a
fine-tuned BERT with layer attention, 80 epochs, BertAdam + ULMFiT-sqrt schedule with gradual
unfreezing). Two limitations, both inherent to UDify:
- **BERT family only** — it uses the old `pytorch_pretrained_bert`, so XLM-R (SentencePiece)
  and the ModernBERT encoders cannot be used.
- **GPU up to compute capability 7.5** — torch 1.4 is a CUDA 10.1 build: Colab's T4 works,
  an A100 does not. On Apple Silicon the setup script uses an x86_64 Python under Rosetta,
  which is enough to test the pipeline on CPU.

## Running several jobs at once

`scripts/gpu_queue.sh` starts jobs only when the GPU really has room for them, so a run
cannot die with CUDA out of memory. Jobs come from stdin, one `<name> <command ...>` per line:

```bash
for sd in 8446 8447 8448; do
  echo "udify_dudu_$sd .venv-udify/bin/python src/run_udify.py run --models berturk \
        --train-sets dudu --cv 0 --seeds $sd"
done > jobs.txt
bash scripts/gpu_queue.sh --slots 4 --need 11000 --log-dir /content/qlogs < jobs.txt
```

`--need` is how much free memory (MiB) one run of that parser wants: STEPS ~8000,
UDify ~11000, MaChAmp ~8000, UDPipe 2 ~6000. The runners skip finished runs, so the same
queue can be started again after a crash or a disconnect.
