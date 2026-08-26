# Multimodal Visual Question Answering (Image + Text)

A compact, reproducible implementation of the **second machine-test task**:

- Image-only baseline
- Text-only baseline
- Multimodal late-fusion model
- 70/10/20 stratified split
- Test Top-1 Accuracy (%)
- 4-option multiple-choice VQA
- FastAPI `POST /v1/qa`
- Reproducible configuration and saved checkpoints

## Important dataset note

The assignment defines each record as an image, question, four options and one correct option. The public VQA v2 dataset is originally **open-ended**, so this repository includes a preparation script that converts an accessible VQA validation-derived dataset into the required four-option format.

Default source:

`SIS-2024-spring/coco_vqa_small_dataset`

This is a small VQA dataset derived from the Graphcore VQA validation data. It contains 1,169 training and 100 validation examples and is convenient for a 24-hour machine test. The script combines the available examples and creates a new stratified 70/10/20 split. See the report for the exact split actually produced.

For a production-scale run, replace the prepared JSONL with a larger VQA v2-derived dataset.

## Architecture

```text
Image -----------------> CLIP image encoder -----> image embedding --\
                                                                       \
Question + 4 options -> MiniLM text encoder -----> text embedding ----> Fusion MLP -> 4 logits
```

The encoders are frozen by default to make the experiment practical on a single GPU/CPU. The trainable heads use AdamW and CrossEntropyLoss.

## Project structure

```text
vqa_multimodal/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits.json
├── checkpoints/
├── outputs/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── prepare_data.py
│   ├── dataset.py
│   ├── encoders.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
├── app.py
├── config.yaml
├── requirements.txt
├── run.sh
└── README.md
```

## 1. Environment

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Prepare the data

```bash
python -m src.prepare_data
```

This downloads the public source through Hugging Face Datasets, creates four choices per question, and writes:

```text
data/processed/all.jsonl
data/processed/train.jsonl
data/processed/val.jsonl
data/processed/test.jsonl
data/splits.json
```

The script never uses the test split for training.

## 3. Train all three models

```bash
python -m src.train --model image
python -m src.train --model text
python -m src.train --model fusion
```

If you have a CUDA GPU:

```bash
python -m src.train --model fusion --device cuda
```

The best validation checkpoint is written to `checkpoints/`.

## 4. Evaluate

```bash
python -m src.evaluate --model image
python -m src.evaluate --model text
python -m src.evaluate --model fusion
```

Results are written to:

```text
outputs/image_metrics.json
outputs/text_metrics.json
outputs/fusion_metrics.json
```

The test set is used only here.

## 5. Inference from command line

```bash
python -m src.infer \
  --image path/to/image.jpg \
  --question "What color is the car?" \
  --options '["red","blue","black","white"]' \
  --model fusion
```

## 6. FastAPI

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

Request:

```bash
curl -X POST "http://localhost:8000/v1/qa" \
  -F "image=@path/to/image.jpg" \
  -F "question=What color is the car?" \
  -F 'options=["red","blue","black","white"]'
```

Response:

```json
{
  "predicted_index": 2,
  "predicted_option": "black",
  "confidence": 0.83
}
```

The confidence is the softmax probability of the selected option.

## 7. What to report

Do not invent results. Run the evaluation scripts and copy the actual values into your final report.

Recommended table:

| Model | Top-1 Accuracy (%) |
|---|---:|
| Image-only | XX.XX |
| Text-only | XX.XX |
| Fusion | XX.XX |

Also discuss:

- whether fusion improves over the best unimodal model
- limitations of the small dataset
- frozen encoders
- random distractor options
- possible improvements with a larger VQA v2 training set and fine-tuning

## 8. Reproducibility

The seed is fixed in `config.yaml`, the split is saved to `data/splits.json`, and the model configuration is stored in the same file.

## 9. GitHub

Do not commit:

- `.venv/`
- `data/raw/`
- downloaded model caches
- large checkpoints if your repository limit is exceeded

Commit the source code, configuration, README, and small JSON metadata. If the checkpoint is too large for GitHub, use Git LFS or provide a release/artifact link.

## 10. Assignment alignment

The implementation follows the assignment's required structure:

- image-only baseline
- text-only baseline
- late fusion
- CrossEntropyLoss
- AdamW
- 70/10/20 split
- test Top-1 Accuracy
- FastAPI `/v1/qa`
- reproducible splits
- README and metrics

