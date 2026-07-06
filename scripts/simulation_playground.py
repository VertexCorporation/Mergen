import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set


# Fix relative imports when launched as `python scripts/simulation_playground.py`.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import MergenBrain_v7 as DigitalBrain


DEFAULT_DATA_PATH = "./data/simulation_texts.txt"
DEFAULT_DREAM_CYCLES = 10
MAX_DREAM_CYCLES = 1000
DEFAULT_EVAL_THRESHOLD = 0.35


STOPWORDS = {
    "bir", "bu", "ve", "ile", "için", "icin", "gibi", "olan", "olarak",
    "daha", "çok", "cok", "az", "en", "da", "de", "mi", "mı", "mu", "mü",
    "the", "and", "or", "of", "to", "in", "is", "are",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mergen experience training harness",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help=f"Experience text file path (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N non-empty experiences.",
    )
    parser.add_argument(
        "--no-dream",
        action="store_true",
        help="Skip Dream consolidation at the end.",
    )
    parser.add_argument(
        "--dream-cycles",
        type=int,
        default=DEFAULT_DREAM_CYCLES,
        help=f"Dream cycles to run, 1..{MAX_DREAM_CYCLES} (default: {DEFAULT_DREAM_CYCLES}).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between experiences (default: 0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned actions without initializing Mergen or writing state.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip post-training evaluation probes.",
    )
    parser.add_argument(
        "--eval-top-k",
        type=int,
        default=3,
        help="Number of KB/RAG candidates to inspect per probe (default: 3).",
    )
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=DEFAULT_EVAL_THRESHOLD,
        help=f"Token-overlap pass threshold for KB/RAG probes (default: {DEFAULT_EVAL_THRESHOLD}).",
    )
    parser.add_argument(
        "--strict-eval",
        action="store_true",
        help="Return a non-zero exit code if post-training evaluation fails.",
    )
    parser.add_argument(
        "--bridge-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for Hebbian-RAG bridge updates before saving (default: 30).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Optional[str]:
    if args.limit is not None and args.limit < 1:
        return "--limit must be >= 1."
    if args.delay < 0:
        return "--delay must be >= 0."
    if args.bridge_timeout < 0:
        return "--bridge-timeout must be >= 0."
    if not 1 <= args.dream_cycles <= MAX_DREAM_CYCLES:
        return f"--dream-cycles must be between 1 and {MAX_DREAM_CYCLES}."
    if args.eval_top_k < 1:
        return "--eval-top-k must be >= 1."
    if not 0.0 <= args.eval_threshold <= 1.0:
        return "--eval-threshold must be between 0.0 and 1.0."
    return None


def load_experiences(path: Path, limit: Optional[int]) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Experience file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Experience path is not a file: {path}")

    with path.open("r", encoding="utf-8") as f:
        experiences = [line.strip() for line in f if line.strip()]

    if limit is not None:
        experiences = experiences[:limit]

    if not experiences:
        raise ValueError(f"No non-empty experiences found in: {path}")

    return experiences


def print_dry_run(args: argparse.Namespace, path: Path, experiences: List[str]) -> int:
    print("[Playground] DRY RUN - no brain init, no writes, no RAG, no Dream.")
    print(f"[Playground] Data source: {path}")
    print(f"[Playground] Experiences selected: {len(experiences)}")
    print("[Playground] Planned actions per experience:")
    print("  1. KB/Hebbian learning via mergen.brain.learn_from_text")
    print("  2. Biological perception via mergen.limbic.respond")
    print("  3. Optional RAG indexing and Hebbian-RAG bridge update")
    if args.no_dream:
        print("[Playground] Dream: skipped (--no-dream)")
    else:
        print(f"[Playground] Dream: planned with {args.dream_cycles} cycles")
    if args.no_eval:
        print("[Playground] Evaluation: skipped (--no-eval)")
    else:
        print(
            "[Playground] Evaluation: planned "
            f"(top_k={args.eval_top_k}, threshold={args.eval_threshold:.2f})"
        )
    print(f"[Playground] Hebbian-RAG bridge wait timeout: {args.bridge_timeout:.1f}s")
    print()

    for i, text in enumerate(experiences, 1):
        print(f"  [{i}] {text}")
    return 0


def make_intent_report(source: str) -> Dict:
    return {
        "primary_intent": "EXPERIENCE",
        "confidence_score": 0.9,
        "sentiment": {"sentiment_score": 0.0, "excitement": 0.0},
        "subject": source,
    }


def learn_experience(mergen: DigitalBrain, text: str, source: str) -> Dict:
    return mergen.brain.learn_from_text(
        text=text,
        vocab=mergen.vocab,
        intent_report=make_intent_report(source),
        learning_rate=0.02,
        reward=1.0,
    )


def index_rag_batch(mergen: DigitalBrain, experiences: List[str], source: str) -> int:
    rag = getattr(mergen, "rag", None)
    if rag is None or not getattr(rag, "ready", False):
        return 0
    return rag.index_texts(texts=experiences, source=source)


def update_hebbian_bridge(
    mergen: DigitalBrain,
    experiences: List[str],
    source: str,
    timeout: float,
) -> int:
    bridge = getattr(mergen, "_hebb_bridge", None)
    if bridge is None:
        return 0
    before = getattr(bridge, "update_count", 0)
    wait_timeout = None if timeout == 0 else timeout
    updated = bridge.update_from_batch(
        experiences,
        source=source,
        reward=1.0,
        wait=True,
        timeout=wait_timeout,
    )
    if isinstance(updated, int):
        return max(0, updated)
    after = getattr(bridge, "update_count", before)
    return max(0, after - before)


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]


def token_set(text: str) -> Set[str]:
    return set(tokenize(text))


def overlap_score(expected: Set[str], candidate_text: str) -> float:
    if not expected:
        return 0.0
    candidate = token_set(candidate_text)
    if not candidate:
        return 0.0
    return len(expected & candidate) / len(expected)


def probe_query(text: str) -> str:
    tokens = tokenize(text)
    if not tokens:
        return text
    return " ".join(tokens[: min(5, len(tokens))])


def best_kb_match(mergen: DigitalBrain, expected: Set[str], top_k: int) -> Dict:
    facts = getattr(mergen.brain, "knowledge_base", [])
    scored = []
    for fact in facts:
        fact_text = fact.get("text", "")
        score = overlap_score(expected, fact_text)
        if score > 0:
            scored.append({"text": fact_text, "score": score})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[0] if scored[:top_k] else {"text": "", "score": 0.0}


def best_rag_match(mergen: DigitalBrain, query: str, expected: Set[str], source: str, top_k: int) -> Dict:
    rag = getattr(mergen, "rag", None)
    if rag is None or not getattr(rag, "ready", False):
        return {"status": "skipped", "text": "", "score": 0.0}

    hits = rag.search(query=query, top_k=top_k, source_filter=source, min_relevance=0.0)
    if not hits:
        hits = rag.search(query=query, top_k=top_k, min_relevance=0.0)

    scored = []
    for hit in hits:
        hit_text = hit.get("text", "")
        scored.append({"text": hit_text, "score": overlap_score(expected, hit_text)})
    scored.sort(key=lambda item: item["score"], reverse=True)
    if not scored:
        return {"status": "ready", "text": "", "score": 0.0}
    best = scored[0]
    best["status"] = "ready"
    return best


def evaluate_training(
    mergen: DigitalBrain,
    experience_reports: List[Dict],
    source: str,
    top_k: int,
    threshold: float,
) -> Dict:
    probes = []
    for report in experience_reports:
        text = report["text"]
        expected = token_set(text)
        query = probe_query(text)
        kb = best_kb_match(mergen, expected, top_k)
        rag = best_rag_match(mergen, query, expected, source, top_k)
        limbic_score = overlap_score(expected, report.get("fired_concepts", ""))

        probes.append({
            "query": query,
            "expected": text,
            "kb_score": kb["score"],
            "kb_pass": kb["score"] >= threshold,
            "rag_score": rag["score"],
            "rag_status": rag["status"],
            "rag_pass": rag["status"] == "skipped" or rag["score"] >= threshold,
            "limbic_score": limbic_score,
            "limbic_pass": limbic_score > 0.0,
        })

    total = len(probes)
    kb_passes = sum(1 for probe in probes if probe["kb_pass"])
    rag_ready = [probe for probe in probes if probe["rag_status"] != "skipped"]
    rag_passes = sum(1 for probe in rag_ready if probe["rag_pass"])
    limbic_passes = sum(1 for probe in probes if probe["limbic_pass"])

    kb_rate = kb_passes / total if total else 0.0
    rag_rate = rag_passes / len(rag_ready) if rag_ready else None
    limbic_rate = limbic_passes / total if total else 0.0

    passed = kb_rate >= 0.8 and (rag_rate is None or rag_rate >= 0.8)
    return {
        "status": "passed" if passed else "attention",
        "probe_count": total,
        "kb_passes": kb_passes,
        "kb_rate": kb_rate,
        "rag_passes": rag_passes,
        "rag_total": len(rag_ready),
        "rag_rate": rag_rate,
        "limbic_passes": limbic_passes,
        "limbic_rate": limbic_rate,
        "threshold": threshold,
        "probes": probes,
    }


def save_persistent_state(mergen: DigitalBrain) -> Dict[str, bool]:
    brain_path = getattr(mergen.config, "MX_KNOWLEDGE_PATH", "./mergen_knowledge.mx")
    result = {
        "brain_saved": mergen.brain.save(brain_path),
        "limbic_saved": False,
    }

    limbic = getattr(mergen, "limbic", None)
    if limbic is not None:
        result["limbic_saved"] = limbic.save_state()

    return result


def run_dream_if_enabled(mergen: DigitalBrain, args: argparse.Namespace) -> Dict:
    if args.no_dream:
        return {"status": "skipped"}

    response = mergen._run_dream_consolidation(cycles=args.dream_cycles)
    return {
        "status": "completed" if "tamamlandi" in response else "error",
        "cycles": args.dream_cycles,
        "response": response,
    }


def print_summary(summary: Dict) -> None:
    print("\n[Playground] Training summary")
    print(f"  Total experiences:     {summary['total_experiences']}")
    print(f"  Processed experiences: {summary['processed_experiences']}")
    print(f"  Failed experiences:    {summary['failed_experiences']}")
    print(f"  Words learned:         {summary['words_learned']}")
    print(f"  KB facts:              {summary['kb_facts']}")
    print(f"  RAG indexed:           {summary['rag_indexed']}")
    print(f"  Hebbian bridge updates:{summary['hebbian_bridge_updates']}")
    print(f"  Last fired concepts:   {summary['last_fired_concepts'] or '-'}")
    print(f"  Brain saved:           {summary['brain_saved']}")
    print(f"  Limbic .mx saved:      {summary['limbic_saved']}")

    dream = summary["dream"]
    if dream["status"] == "skipped":
        print("  Dream:                skipped")
    else:
        print(f"  Dream:                {dream['status']} ({dream.get('cycles', 0)} cycles)")

    evaluation = summary.get("evaluation")
    if evaluation is None:
        print("  Evaluation:           skipped")
    else:
        rag_text = "skipped"
        if evaluation["rag_rate"] is not None:
            rag_text = f"{evaluation['rag_passes']}/{evaluation['rag_total']} ({evaluation['rag_rate']:.0%})"
        print("  Evaluation:")
        print(f"    Status:             {evaluation['status']}")
        print(f"    KB probes:          {evaluation['kb_passes']}/{evaluation['probe_count']} ({evaluation['kb_rate']:.0%})")
        print(f"    RAG probes:         {rag_text}")
        print(f"    Limbic overlap:     {evaluation['limbic_passes']}/{evaluation['probe_count']} ({evaluation['limbic_rate']:.0%})")


def main() -> int:
    args = parse_args()
    validation_error = validate_args(args)
    if validation_error:
        print(f"[Playground] Argument error: {validation_error}")
        return 2

    data_path = Path(args.data)
    try:
        experiences = load_experiences(data_path, args.limit)
    except Exception as e:
        print(f"[Playground] Error: {e}")
        return 1

    if args.dry_run:
        return print_dry_run(args, data_path, experiences)

    print("[Playground] Waking up Mergen in Experience Training Mode...")
    try:
        mergen = DigitalBrain(verbose=False)
    except Exception as exc:
        print(f"[Playground] Error initializing Mergen: {exc}")
        return 1

    if not hasattr(mergen, "limbic") or mergen.limbic is None:
        print("[Playground] Error: Biological Core (Limbic Layer) is not active.")
        return 1

    source = data_path.stem
    words_learned = 0
    failed = 0
    fired_concepts: List[str] = []
    experience_reports: List[Dict] = []

    print(f"[Playground] Data source: {data_path}")
    print(f"[Playground] Experiences to process: {len(experiences)}\n")

    for i, text in enumerate(experiences, 1):
        print(f"--- Experience {i}/{len(experiences)} ---")
        print(f"Input: {text}")
        try:
            learn_result = learn_experience(mergen, text, source)
            words_learned += int(learn_result.get("words_learned", 0))
            print(f"Learned words: {learn_result.get('words_learned', 0)}")

            response = mergen.limbic.respond(user_input=text, max_attempts=1)
            fired = ""
            if mergen.limbic.last_thought:
                fired = mergen.limbic.last_thought
                fired_concepts.append(fired)
                print(f"Fired concepts: {fired}")
            print(f"Broca response: {response}")
            experience_reports.append({
                "text": text,
                "learned_words": int(learn_result.get("words_learned", 0)),
                "fired_concepts": fired,
            })
        except Exception as e:
            failed += 1
            print(f"[Playground] Experience failed: {e}")

        if args.delay > 0:
            time.sleep(args.delay)
        print("-" * 50)

    rag_indexed = 0
    hebb_updates = 0
    try:
        rag_indexed = index_rag_batch(mergen, experiences, source)
    except Exception as e:
        print(f"[Playground] RAG indexing failed: {e}")

    try:
        hebb_updates = update_hebbian_bridge(
            mergen=mergen,
            experiences=experiences,
            source=source,
            timeout=args.bridge_timeout,
        )
    except Exception as e:
        print(f"[Playground] Hebbian-RAG bridge update failed: {e}")

    evaluation = None
    if not args.no_eval:
        try:
            evaluation = evaluate_training(
                mergen=mergen,
                experience_reports=experience_reports,
                source=source,
                top_k=args.eval_top_k,
                threshold=args.eval_threshold,
            )
        except Exception as e:
            evaluation = {
                "status": "error",
                "probe_count": 0,
                "kb_passes": 0,
                "kb_rate": 0.0,
                "rag_passes": 0,
                "rag_total": 0,
                "rag_rate": None,
                "limbic_passes": 0,
                "limbic_rate": 0.0,
                "threshold": args.eval_threshold,
                "probes": [],
                "error": str(e),
            }
            print(f"[Playground] Evaluation failed: {e}")

    save_result = save_persistent_state(mergen)
    dream_result = run_dream_if_enabled(mergen, args)

    summary = {
        "total_experiences": len(experiences),
        "processed_experiences": len(experiences) - failed,
        "failed_experiences": failed,
        "words_learned": words_learned,
        "kb_facts": mergen.brain.knowledge_size(),
        "rag_indexed": rag_indexed,
        "hebbian_bridge_updates": hebb_updates,
        "last_fired_concepts": fired_concepts[-1] if fired_concepts else "",
        "brain_saved": save_result["brain_saved"],
        "limbic_saved": save_result["limbic_saved"],
        "dream": dream_result,
        "evaluation": evaluation,
    }
    print_summary(summary)

    if dream_result["status"] == "error":
        print("\n[Playground] Dream response:")
        print(dream_result["response"])
        return 1

    if args.strict_eval and evaluation is not None and evaluation["status"] != "passed":
        return 1

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
