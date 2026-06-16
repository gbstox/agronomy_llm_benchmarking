"""Harden text multiple-choice questions by expanding them to 8 options.

For each non-image question, a strong "generator" model proposes additional
plausible-but-incorrect distractors, and a separate "verifier" model audits them
to reject any that could actually be correct. The original options are kept
(they are human-authored and vetted); accepted distractors are added and all
options are reshuffled to NUM_OPTIONS.

Scoring is self-contained per result file, so changing the central questions
file does not retroactively affect already-run models — only models that are
re-run will see the new options.

    .venv/bin/python harden_text_questions.py --limit 5      # sample/test
    .venv/bin/python harden_text_questions.py                # all questions
"""
import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config

PROJECT_ROOT = Path(__file__).resolve().parent
QUESTIONS_FILE = (PROJECT_ROOT / config.BENCHMARK_QUESTIONS_FILE).resolve()
IMAGE_CATEGORY = config.IMAGE_QUESTION_CATEGORY
NUM_OPTIONS = 8
OPTION_KEYS = ["a", "b", "c", "d", "e", "f", "g", "h"]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GENERATOR_MODEL = "openai/gpt-5.5"
VERIFIER_MODEL = "google/gemini-3.1-pro-preview"


def _api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    for line in (PROJECT_ROOT / ".env").read_text().splitlines():
        if line.strip().startswith("OPENROUTER_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OPENROUTER_API_KEY not found")


API_KEY = _api_key()


def _chat(model, system, user, max_tokens=1200, retries=5):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(
            OPENROUTER_URL, data=body,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read().decode())
            return d["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            last = e
            time.sleep(min(30, 3 * (2 ** attempt)))
    raise last


def _extract_json_array(text):
    if not text:
        return []
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [str(x).strip() for x in arr if str(x).strip()]
    except json.JSONDecodeError:
        return []


def _generate_distractors(question, options, correct_text, k):
    system = (
        "You write challenging but fair distractor options for agronomy multiple-choice exam "
        "questions. Distractors must be plausible and topically on-point, yet unambiguously INCORRECT."
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"EXISTING OPTIONS:\n{json.dumps(options, ensure_ascii=False)}\n\n"
        f"CORRECT ANSWER:\n{correct_text}\n\n"
        f"Write {k} ADDITIONAL answer options that are:\n"
        f"- plausible and clearly relevant to the question,\n"
        f"- definitely INCORRECT (not correct, not a paraphrase/synonym of the correct answer, not partially correct),\n"
        f"- distinct from each other and from the existing options,\n"
        f"- similar in length, style and specificity to the existing options.\n"
        f"Return ONLY a JSON array of {k} strings."
    )
    return _extract_json_array(_chat(GENERATOR_MODEL, system, user))


def _verify_distractors(question, correct_text, candidates):
    """Returns the subset of candidates that the verifier deems safely incorrect."""
    if not candidates:
        return []
    system = (
        "You audit candidate distractor options for agronomy multiple-choice questions. "
        "Your job is to protect benchmark integrity by removing any candidate that could be "
        "considered correct or as-correct-as the stated correct answer."
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"CORRECT ANSWER:\n{correct_text}\n\n"
        f"CANDIDATE DISTRACTORS:\n{json.dumps(candidates, ensure_ascii=False)}\n\n"
        "Return ONLY a JSON array containing exactly the candidates that are SAFELY INCORRECT "
        "(should be kept as distractors). Omit any that might be correct, ambiguous, or a synonym "
        "of the correct answer. Copy kept strings verbatim."
    )
    kept = _extract_json_array(_chat(VERIFIER_MODEL, system, user))
    cand_set = {c.lower(): c for c in candidates}
    return [cand_set[k.lower()] for k in kept if k.lower() in cand_set]


def harden_question(q, rng):
    options = q.get("answer_options", {})
    if not isinstance(options, dict) or len(options) >= NUM_OPTIONS:
        return None
    correct_key = q.get("correct_answer")
    correct_text = options.get(correct_key)
    if correct_text is None:
        return None
    existing_values = list(options.values())
    needed = NUM_OPTIONS - len(existing_values)

    accepted = []
    existing_lower = {v.lower() for v in existing_values}
    for _ in range(3):  # a few rounds to reach the needed count after verification
        if len(accepted) >= needed:
            break
        cand = _generate_distractors(q["question"], existing_values, correct_text, needed - len(accepted) + 1)
        cand = [c for c in cand if c.lower() not in existing_lower and c.lower() not in {a.lower() for a in accepted}
                and c.lower() != correct_text.lower()]
        verified = _verify_distractors(q["question"], correct_text, cand)
        accepted.extend(verified)

    accepted = accepted[:needed]
    if not accepted:
        return None

    all_values = existing_values + accepted
    rng.shuffle(all_values)
    new_options = {key: val for key, val in zip(OPTION_KEYS, all_values)}
    new_correct = next(key for key, val in new_options.items() if val == correct_text)
    q["answer_options"] = new_options
    q["correct_answer"] = new_correct
    q["_hardened_added"] = len(accepted)
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Only process N questions (0 = all).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true", help="Print samples; do not write the questions file.")
    args = parser.parse_args()

    bench = json.load(open(QUESTIONS_FILE))
    mc = bench["multiple_choice"]
    targets = []
    for category, qs in mc.items():
        if category == IMAGE_CATEGORY or not isinstance(qs, list):
            continue
        for q in qs:
            if isinstance(q, dict) and len(q.get("answer_options", {})) < NUM_OPTIONS:
                targets.append(q)
    if args.limit:
        targets = targets[: args.limit]

    print(f"Hardening {len(targets)} questions to {NUM_OPTIONS} options "
          f"(gen={GENERATOR_MODEL}, verify={VERIFIER_MODEL})...")

    rng = random.Random(args.seed)
    done = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(harden_question, q, random.Random(rng.random())): q for q in targets}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                res = fut.result()
                if res is None:
                    fail += 1
                else:
                    done += 1
            except Exception as e:
                fail += 1
                print(f"  error: {type(e).__name__} {e}")
            if i % 25 == 0:
                print(f"  {i}/{len(targets)} processed (hardened={done}, failed={fail})")

    print(f"\nHardened {done} questions; {fail} unchanged/failed.")

    if args.dry_run:
        for q in targets[:5]:
            print("\n---", q.get("id"))
            print("Q:", q["question"][:120])
            print("correct:", q["correct_answer"], "->", q["answer_options"][q["correct_answer"]])
            for k, v in q["answer_options"].items():
                print(f"   {k}: {v}")
        print("\n[dry-run] questions file NOT written.")
        return

    json.dump(bench, open(QUESTIONS_FILE, "w"), indent=2)
    print(f"Wrote hardened questions to {QUESTIONS_FILE}")


if __name__ == "__main__":
    main()
