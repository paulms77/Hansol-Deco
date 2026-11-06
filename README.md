# Hansol Deco Season 3 — Construction Accident Prevention & Response Generation

RAG pipeline for the DACON **"건설공사 사고 예방 및 대응책 생성"** generative-AI
competition: given the circumstances of a construction-site accident, generate
the recurrence-prevention measures and follow-up action plan.

| | |
| --- | --- |
| **Competition** | [건설공사 사고 예방 및 대응책 생성 : 한솔데코 시즌3 생성 AI 경진대회](https://dacon.io/competitions/official/236455/overview/description) |
| **Host** | Hansol Deco · DACON |
| **Period** | 17 Feb 2025 – 24 Mar 2025 |
| **Task** | Korean text generation, grounded in retrieved safety guidelines |
| **Metric** | Jaccard similarity (lexical) + S-BERT cosine similarity (semantic) |
| **Environment** | Google Colab (T4) and Kaggle notebooks — free tier only |

---

## Table of contents

- [The problem](#the-problem)
- [Why RAG instead of a large LLM](#why-rag-instead-of-a-large-llm)
- [Data](#data)
- [Evaluation](#evaluation)
- [Pipeline](#pipeline)
- [Notebooks](#notebooks)
- [Setup](#setup)
- [Usage](#usage)
- [Notes from building this](#notes-from-building-this)
- [Repository layout](#repository-layout)

---

## The problem

Each record describes one real construction accident — weather, temperature,
humidity, type of construction, floor area, floor information, human injury,
property damage, trade, object involved, work process, location, body part, and
a free-text accident cause. The target is the free-text field
`재발방지대책 및 향후조치계획` ("recurrence prevention measures and future action
plan").

So this is not classification. It is conditional generation that has to produce
plausible, specific, *domain-correct* Korean safety prose — which is exactly
where a model with no construction-safety knowledge falls over.

## Why RAG instead of a large LLM

The obvious approach is to fine-tune a large Korean LLM. This project
deliberately does not, for two reasons.

**Compute.** The whole thing had to run on Colab's free T4 and Kaggle notebooks.
A 10B+ model with meaningful context does not fit that budget, and anything that
does not fit cannot be iterated on.

**Grounding.** The answer has to be practical and correct, not fluent. The
competition ships 104 official construction-safety guideline PDFs — the domain
knowledge is *already written down*. Retrieving the right passage and conditioning
on it gets closer to a correct answer than asking a general model to recall
safety procedure from its weights.

That reframes the problem from "train a bigger generator" to "retrieve the right
evidence", which is both cheaper and more accurate here.

## Data

From [the competition page](https://dacon.io/competitions/official/236455/data):

| File | Contents |
| --- | --- |
| `train.csv` | accident records **with** the target text |
| `test.csv` | accident records without the target |
| `건설안전지침/` | 104 construction-safety guideline PDFs |
| `sample_submission.csv` | ID, generated answer, and `vec_0 … vec_767` embedding columns |

`Dataset/` in this repository holds `test.csv` and `sample_submission.csv` plus a
[data dictionary](Dataset/README.md). `train.csv` and the PDF corpus are not
redistributed — download them from DACON.

### Feature engineering

Two fields are hierarchical strings and get split into three levels each:

```
공사종류:  "건축 / 건축물 / 근린생활시설"
             → 대분류 = 건축
             → 중분류 = 건축물
             → 소분류 = 근린생활시설

사고객체:  same treatment (대분류 / 중분류 / 소분류)
```

This matters for retrieval: matching on `건축` alone is far too coarse, while
matching on the full raw string is too sparse. Three levels give the retriever
something to key on at the right granularity.

## Evaluation

Submissions carry both the generated sentence and its 768-dimensional embedding.
The score combines two similarities against the ground-truth sentence:

- **Jaccard similarity** — lexical overlap of the word sets
- **S-BERT cosine similarity** — semantic closeness of the embeddings

```python
def cosine_similarity(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return dot / (na * nb) if na != 0 and nb != 0 else 0

def jaccard_similarity(text1, text2):
    s1, s2 = set(text1.split()), set(text2.split())
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0
```

> **The embedding model is fixed by the rules.** Ground-truth vectors are
> produced with `jhgan/ko-sbert-sts`, and the submitted embedding must come from
> the same model and must describe the same text that is scored for Jaccard.
> Using a different encoder for the submission silently destroys the score.

Note the tension the metric creates: Jaccard rewards reusing the exact vocabulary
of the reference, while cosine rewards meaning. A fluent paraphrase scores badly
on the first; a keyword salad scores badly on the second.

## Pipeline

```
 train.csv ─┐
            ├─► field splitting ─► query construction ─┐
 test.csv  ─┘                                          │
                                                       ▼
 104 safety PDFs ─► load ─► clean ─► split ─► embed ─► hybrid retrieval
                                                       │   BM25 (lexical)
                                                       │   FAISS (semantic)
                                                       ▼
                                           threshold filter (BM25 + cosine)
                                                       ▼
                                             retrieved evidence (top-k = 3)
                                                       ▼
                                         generation (MT5 / Llama, [INST])
                                                       ▼
                                   answer  ─►  ko-sbert-sts  ─►  submission.csv
```

### 1. Load

PDF loader choice is not incidental. `fitz` exposes little beyond page numbers,
whereas `langchain`'s `PdfReader` handles Korean encoding well and is stable on
throughput — so `PdfReader` it is.

### 2. Clean

Safety guideline PDFs are full of material that is noise for retrieval: images,
special characters, stray whitespace and newlines, appendices, and repeated
boilerplate such as `KOSHA GUIDE C552015` headers. All of it is stripped so that
only text relevant to accident-cause analysis survives.

### 3. Split

`RecursiveCharacterTextSplitter` chunks the cleaned documents so each piece can
be embedded and retrieved independently.

### 4. Embed

`SentenceTransformer("jhgan/ko-sbert-nli")` for the retrieval corpus — a Korean
sentence encoder, unlike the English-only defaults, which matters when both the
query and the corpus are Korean.

> The submission embedding is a separate step and **must** use
> `jhgan/ko-sbert-sts` per the rules.

### 5. Store and retrieve — hybrid

Two indexes over the same chunks:

- **BM25** (`rank_bm25`) for keyword matching — good at catching exact domain
  terms like 아웃트리거 or 타설작업
- **FAISS** over the sentence embeddings for semantic matching — good at catching
  paraphrases the keyword index misses

Scores from both are combined and only chunks above a **threshold** are kept.
The filter is the useful part: without it, top-k always returns *something*, and
irrelevant evidence actively degrades the generated answer. Retrieving nothing is
better than retrieving noise.

Similarity computation over the full corpus is parallelised (multiprocessing,
with `joblib` caching of retrieved results) — on free-tier hardware, re-running
retrieval from scratch on every iteration is the main time sink.

### 6. Generate

Two generators were tried:

- **`google/mt5-small`** — multilingual T5, handles Korean, small enough for the
  compute budget

  ```python
  def generate_prevention_plan(query_text, retrieved_texts, max_length=200):
      input_text = f"사고 원인: {query_text} \n 관련 문서: {' '.join(retrieved_texts)}"
      inputs = tokenizer(input_text, return_tensors="pt",
                         truncation=True, padding=True, max_length=512)
      output_ids = model.generate(**inputs, max_length=max_length)
      return tokenizer.decode(output_ids[0], skip_special_tokens=True)
  ```

- **Llama-family instruct models** — where wrapping the prompt in `[INST] … [/INST]`
  turned out to matter a great deal for output quality

A **retrieval-only** submission is also produced, which simply returns the
prevention plan of the most similar historical accident. It is a strong baseline:
when a small generator produces vague text, a real sentence written by a real
engineer about a genuinely similar accident often scores better on both halves of
the metric.

## Notebooks

The two notebooks are the same pipeline at two stages, in two environments.
Reading them side by side shows what changed.

| | [`01_retrieval_colab.ipynb`](notebooks/01_retrieval_colab.ipynb) | [`02_rag_generation_kaggle.ipynb`](notebooks/02_rag_generation_kaggle.ipynb) |
| --- | --- | --- |
| Environment | Google Colab (Drive mount, `/gdrive/MyDrive/open`) | Kaggle (`/kaggle/input/opendata`) |
| Cells | 68 | 78 |
| Retrieval query fields | `인적사고`, `작업프로세스`, `사고원인` | `인적사고`, `사고원인` |
| Retrieval | ✅ BM25 + FAISS hybrid | ✅ same |
| Retrieval-only submission | ✅ `submission.csv` | ✅ `rag_submission.csv` |
| Generation stage | partial (MT5 experiments) | ✅ full, `final_submission.csv` |
| Llama `[INST]` prompting | — | ✅ |

The narrowing of the retrieval query from three fields to two is the substantive
change: `작업프로세스` ("work process") turned out to add more noise than signal,
pulling in accidents that shared a work stage but had unrelated causes.

## Setup

```bash
git clone https://github.com/paulms77/Hansol-Deco.git
cd Hansol-Deco

pip install transformers peft accelerate bitsandbytes datasets huggingface_hub
pip install PyPDF2 faiss-cpu sumy rank_bm25 sentence-transformers langchain
```

Then download `train.csv` and the `건설안전지침/` PDF folder from the
[competition data page](https://dacon.io/competitions/official/236455/data) and
place them next to `Dataset/`:

```
Dataset/
├── train.csv              ← from DACON
├── test.csv               ✓ included
├── sample_submission.csv  ✓ included
└── 건설안전지침/           ← from DACON (104 PDFs)
```

## Usage

Both notebooks run top to bottom. Set the data root first:

```python
# 01_retrieval_colab.ipynb — Colab
from google.colab import drive
drive.mount('/gdrive')
train_dir = '/gdrive/MyDrive/open/train.csv'
PDF_DIR   = '/gdrive/MyDrive/open/건설안전지침'

# 02_rag_generation_kaggle.ipynb — Kaggle
train_dir = '/kaggle/input/opendata/train.csv'
PDF_DIR   = '/kaggle/input/adddata/건설안전지침'
```

Tunable knobs:

| Parameter | Default | Effect |
| --- | --- | --- |
| `top_k` | `3` | retrieved chunks per query — raising it adds context and noise together |
| similarity threshold | — | drop weak evidence; the single highest-leverage setting |
| `max_length` | `200` | generated answer length |
| `needed_columns` | see above | which fields form the retrieval query |

Outputs, written UTF-8-SIG so Excel opens them correctly:

- `submission.csv` / `rag_submission.csv` — retrieval-only
- `final_submission.csv` — generated answers

Remember to re-encode the final text column with `jhgan/ko-sbert-sts` before
submitting.

## Notes from building this

**The labels are broad, not rule-generated.** The target text reads as a
generically-phrased measure attached to a specific cause, rather than something
produced by a consistent template. That shapes the whole approach: there is no
rule to reverse-engineer, so retrieving a semantically close real example beats
trying to synthesise from structure.

**Quantitative metrics for generated text are a poor proxy for quality.** There
is no settled way to score generated Korean prose, and Jaccard + cosine is a
compromise, not a measure of usefulness. Closing the gap between what the metric
rewards and what a good answer looks like is where most of the leverage lives —
e.g. sampling several candidate answers and selecting among them with your own
quality criterion, rather than taking a single greedy decode.

**Retrieval quality dominates generation quality.** On this compute budget, time
spent on the threshold and the hybrid weighting paid off consistently more than
time spent on the generator.

## Repository layout

```
Dataset/
├── README.md              data dictionary and metric definition
├── test.csv               test records
└── sample_submission.csv  submission format (ID + text + 768-dim vector)

notebooks/
├── 01_retrieval_colab.ipynb        retrieval pipeline, Colab
└── 02_rag_generation_kaggle.ipynb  retrieval + generation, Kaggle
```

## Related

Season 2 of the same series — a QA service for interior-finishing defects —
is in [Wallpaper-defects-knowledge-chat-service](https://github.com/paulms77/Wallpaper-defects-knowledge-chat-service).
