"""Build image-input identification questions for the agronomy benchmark.

Pulls labeled images from public sources and writes multiple-choice questions
(one correct label + 3 same-domain distractors) into the ``image_identification``
category of the benchmark questions file.

Sources (labels are human-readable ground truth):
  - Disease: PlantVillage GitHub mirror (spMohanty/PlantVillage-Dataset)
  - Pests:   IP102 via the HuggingFace datasets-server (hibana2077/IP102)
  - Weeds:   DeepWeeds (ButterChicken98/deepweeds_clip_corpus_90_10) +
             CottonWeedID15 (alexsenden/cottonweedid15_partitioned)

Multiple images per class are allowed, distributed to maximize class diversity.
Images are saved under ``benchmark_questions/images/<bucket>/`` and referenced by
a path relative to the questions file. Both the images and the questions file are
gitignored, so the benchmark content never reaches the public repository.

    .venv/bin/python build_image_questions.py --per-bucket 50
"""
import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import config

PROJECT_ROOT = Path(__file__).resolve().parent
QUESTIONS_FILE = (PROJECT_ROOT / config.BENCHMARK_QUESTIONS_FILE).resolve()
QUESTIONS_DIR = QUESTIONS_FILE.parent
IMAGES_DIR = QUESTIONS_DIR / config.IMAGE_QUESTIONS_DIR
CATEGORY = config.IMAGE_QUESTION_CATEGORY

OPTION_KEYS = ["a", "b", "c", "d"]

PLANTVILLAGE_API = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"
HF_ROWS = "https://datasets-server.huggingface.co/rows"
HF_INFO = "https://datasets-server.huggingface.co/info"
IP102_DATASET = "hibana2077/IP102"
DEEPWEEDS_DATASET = "ButterChicken98/deepweeds_clip_corpus_90_10"
COTTONWEED_DATASET = "alexsenden/cottonweedid15_partitioned"

USER_AGENT = "agronomy-benchmark-image-builder"


GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def _http_get(url, timeout, retries=5):
    """GET with retry/backoff on rate limits (429) and transient 5xx errors."""
    last_err = None
    for attempt in range(retries):
        headers = {"User-Agent": USER_AGENT}
        if GITHUB_TOKEN and url.startswith("https://api.github.com/"):
            headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                wait = min(30, 3 * (2 ** attempt))
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError as e:
            last_err = e
            time.sleep(min(30, 3 * (2 ** attempt)))
    raise last_err


def _http_json(url):
    return json.loads(_http_get(url, timeout=60).decode("utf-8"))


def _download(url, dest_path):
    data = _http_get(url, timeout=120)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(data)


def _slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _pretty_plantvillage(class_name):
    """'Tomato___Late_blight' -> 'Tomato - Late blight'."""
    if "___" in class_name:
        plant, condition = class_name.split("___", 1)
    else:
        plant, condition = class_name, ""
    plant = plant.replace("_", " ").strip()
    condition = condition.replace("_", " ").strip()
    return f"{plant} - {condition}" if condition else plant


def _clean_ip102(label_name):
    """'1  rice leaf roller' -> 'rice leaf roller'."""
    return re.sub(r"^\s*\d+\s+", "", label_name).strip()


def _build_question(qid, image_rel_path, prompt, correct_label, distractor_pool, rng, source):
    pool = list(set(distractor_pool) - {correct_label})
    distractors = rng.sample(pool, k=min(3, len(pool)))
    options = distractors + [correct_label]
    rng.shuffle(options)
    answer_options = {key: opt for key, opt in zip(OPTION_KEYS, options)}
    correct_key = next(key for key, opt in answer_options.items() if opt == correct_label)
    return {
        "id": qid,
        "image": image_rel_path,
        "question": prompt,
        "answer_options": answer_options,
        "correct_answer": correct_key,
        "source": source,
    }


# ----------------------------------------------------------------------------
# Disease (PlantVillage GitHub) - round-robin across classes for diversity
# ----------------------------------------------------------------------------
def build_disease(n, rng, prompt):
    print("[disease] Listing PlantVillage classes...")
    classes = [item["name"] for item in _http_json(PLANTVILLAGE_API) if item["type"] == "dir"]
    label_pool = [_pretty_plantvillage(c) for c in classes]
    rng.shuffle(classes)

    file_cache = {}
    used = {}
    questions = []
    idx = 0
    while len(questions) < n:
        progressed = False
        for class_name in classes:
            if len(questions) >= n:
                break
            if class_name not in file_cache:
                try:
                    files = _http_json(f"{PLANTVILLAGE_API}/{urllib.parse.quote(class_name)}")
                except Exception as e:
                    print(f"  [disease] list fail {class_name}: {e}")
                    file_cache[class_name] = []
                    used[class_name] = 0
                    continue
                imgs = [f for f in files if f["name"].lower().endswith((".jpg", ".jpeg", ".png"))]
                rng.shuffle(imgs)
                file_cache[class_name] = imgs
                used[class_name] = 0
                time.sleep(0.2)
            imgs = file_cache[class_name]
            if used[class_name] >= len(imgs):
                continue
            pick = imgs[used[class_name]]
            used[class_name] += 1
            try:
                ext = Path(pick["name"]).suffix or ".jpg"
                rel = f"{config.IMAGE_QUESTIONS_DIR}/disease/{_slug(class_name)}_{idx}{ext}"
                _download(pick["download_url"], QUESTIONS_DIR / rel)
            except Exception as e:
                print(f"  [disease] dl fail {class_name}: {e}")
                continue
            label = _pretty_plantvillage(class_name)
            questions.append(_build_question(
                f"img_disease_{idx:03d}", rel, prompt, label, label_pool, rng, "PlantVillage"))
            idx += 1
            progressed = True
        if not progressed:
            break
    print(f"  [disease] built {len(questions)} questions")
    return questions


# ----------------------------------------------------------------------------
# HuggingFace datasets-server helpers
# ----------------------------------------------------------------------------
def _hf_rows(dataset, split, offset, length=100, config_name="default"):
    url = (
        f"{HF_ROWS}?dataset={urllib.parse.quote(dataset)}&config={config_name}"
        f"&split={split}&offset={offset}&length={length}"
    )
    return _http_json(url)


def _hf_label_names(dataset, split, config_name="default"):
    payload = _hf_rows(dataset, split, 0, 1, config_name)
    for feature in payload.get("features", []):
        if feature["type"].get("_type") == "ClassLabel":
            return feature["name"], feature["type"].get("names", [])
    return None, []


def _hf_split_size(dataset, split, config_name="default"):
    try:
        info = _http_json(f"{HF_INFO}?dataset={urllib.parse.quote(dataset)}")
        return info["dataset_info"][config_name]["splits"][split]["num_examples"]
    except Exception:
        return 5000


def _collect_label_rows(dataset, split, n, rng, label_resolver, max_per_label=4):
    """Collect up to n (label, image_src) pairs spanning the whole split.

    HuggingFace splits are typically sorted by class, so a single 100-row window
    covers only one or two labels. We walk evenly-spaced window offsets across the
    entire split (which deterministically reaches every class) and apply a
    per-label cap for diversity. The cap is relaxed on each pass for datasets that
    have fewer classes than n.
    """
    size = max(1, _hf_split_size(dataset, split))
    windows = min(max(20, n), 60)
    stride = max(1, size // windows)
    offsets = list(range(0, size, stride))
    rng.shuffle(offsets)

    collected = []
    seen_src = set()
    per_label = {}
    cap = max_per_label
    for _pass in range(3):
        if len(collected) >= n:
            break
        for offset in offsets:
            if len(collected) >= n:
                break
            try:
                payload = _hf_rows(dataset, split, offset, 100)
            except Exception:
                continue
            rows = payload.get("rows", [])
            rng.shuffle(rows)
            for r in rows:
                row = r["row"]
                label = label_resolver(row)
                if not label or per_label.get(label, 0) >= cap:
                    continue
                img = row.get("image")
                src = img.get("src") if isinstance(img, dict) else None
                if not src or src in seen_src:
                    continue
                seen_src.add(src)
                per_label[label] = per_label.get(label, 0) + 1
                collected.append((label, src))
                if len(collected) >= n:
                    break
            time.sleep(0.35)
        cap += max_per_label  # relax for sources with few classes
    return collected


def build_pests(n, rng, prompt):
    print("[pests] Loading IP102 label vocabulary...")
    _, names = _hf_label_names(IP102_DATASET, "test")
    clean_names = [_clean_ip102(x) for x in names]

    def resolver(row):
        idx = row.get("label")
        if isinstance(idx, int) and 0 <= idx < len(clean_names):
            return clean_names[idx]
        return None

    rows = _collect_label_rows(IP102_DATASET, "test", n, rng, resolver)
    questions = []
    for idx, (label, src) in enumerate(rows):
        try:
            rel = f"{config.IMAGE_QUESTIONS_DIR}/pests/{_slug(label)}_{idx}.jpg"
            _download(src, QUESTIONS_DIR / rel)
            questions.append(_build_question(
                f"img_pest_{idx:03d}", rel, prompt, label, clean_names, rng, "IP102"))
        except Exception as e:
            print(f"  [pests] skip {label}: {type(e).__name__} {e}")
    print(f"  [pests] built {len(questions)} questions")
    return questions


def build_weeds(n, rng, prompt):
    print("[weeds] Collecting weed species from DeepWeeds + CottonWeedID15...")

    # DeepWeeds: human-readable 'species' field (exclude the 'negative' class).
    def deepweeds_resolver(row):
        species = row.get("species")
        if not species or str(row.get("label", "")).lower() == "negative" or species.lower() == "negative":
            return None
        return species

    # CottonWeedID15: ClassLabel index -> name.
    _, cotton_names = _hf_label_names(COTTONWEED_DATASET, "train")

    def cotton_resolver(row):
        idx = row.get("label")
        if isinstance(idx, int) and 0 <= idx < len(cotton_names):
            return cotton_names[idx]
        return None

    half = n // 2
    deep_rows = _collect_label_rows(DEEPWEEDS_DATASET, "train", n - half, rng, deepweeds_resolver)
    cotton_rows = _collect_label_rows(COTTONWEED_DATASET, "train", half, rng, cotton_resolver)

    tagged = [("DeepWeeds", lbl, src) for lbl, src in deep_rows] + \
             [("CottonWeedID15", lbl, src) for lbl, src in cotton_rows]
    rng.shuffle(tagged)

    species_pool = list({lbl for _, lbl, _ in tagged} | set(cotton_names))
    questions = []
    for idx, (source, label, src) in enumerate(tagged):
        try:
            rel = f"{config.IMAGE_QUESTIONS_DIR}/weeds/{_slug(label)}_{idx}.jpg"
            _download(src, QUESTIONS_DIR / rel)
            questions.append(_build_question(
                f"img_weed_{idx:03d}", rel, prompt, label, species_pool, rng, source))
        except Exception as e:
            print(f"  [weeds] skip {label}: {type(e).__name__} {e}")
    print(f"  [weeds] built {len(questions)} questions")
    return questions


def main():
    parser = argparse.ArgumentParser(description="Build image identification benchmark questions.")
    parser.add_argument("--per-bucket", type=int, default=50, help="Questions per source (disease/pests/weeds).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible sampling.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    all_questions = []
    all_questions += build_disease(
        args.per_bucket, rng,
        "The image shows a plant leaf. Which plant and condition (healthy or disease) is shown?")
    all_questions += build_pests(
        args.per_bucket, rng,
        "The image shows an agricultural insect pest. Which pest is shown?")
    all_questions += build_weeds(
        args.per_bucket, rng,
        "The image shows a weed plant. Which weed species is shown?")

    if not all_questions:
        print("No questions built; aborting without modifying the benchmark file.")
        return

    with open(QUESTIONS_FILE, "r") as f:
        benchmark = json.load(f)
    benchmark.setdefault("multiple_choice", {})
    benchmark["multiple_choice"][CATEGORY] = all_questions

    with open(QUESTIONS_FILE, "w") as f:
        json.dump(benchmark, f, indent=2)

    print(f"\nWrote {len(all_questions)} image questions to '{QUESTIONS_FILE}' under '{CATEGORY}'.")
    print(f"Images saved under '{IMAGES_DIR}'.")


if __name__ == "__main__":
    main()
