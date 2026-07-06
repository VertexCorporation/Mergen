import argparse
import copy
import json
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import MergenBrain_v7


DEFAULT_PROBES = [
    "Mergen nedir?",
    "Hebbian öğrenme nedir?",
    "STDP neyi değiştirir?",
    "Kütleçekim nedir?",
    "RAG ne işe yarar?",
    "Dream konsolidasyonu ne yapar?",
    "Sinaps nedir?",
    "Plastisite nedir?",
    "Kuantum parçacık nasıl temsil edilir?",
    "Öğrenme verisi neden tutarlı olmalıdır?",
]

STOPWORDS = {
    "bir", "ve", "ile", "bu", "şu", "o", "de", "da", "mi", "mı", "mu",
    "mü", "ki", "için", "gibi", "olan", "çok", "daha", "en", "ne",
    "nedir", "demek", "neden", "nasıl", "hangi", "kim", "neye", "neyi",
    "what", "why", "how", "who", "where", "when", "is", "are", "the",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Mergen cognitive architecture behavior.")
    parser.add_argument("--probes", default=None, help="Optional UTF-8 text file with one probe per line.")
    parser.add_argument("--logs-dir", default="logs", help="Output directory (default: logs).")
    parser.add_argument("--name", default="cognitive_audit", help="Run name prefix.")
    parser.add_argument("--max-probes", type=int, default=None, help="Limit number of probes.")
    parser.add_argument("--dream-cycles", type=int, default=0, help="If >0, run Dream and repeat baseline.")
    parser.add_argument("--conditions", default="baseline,no_rag,no_limbic,no_semantic,no_hebbian_trace",
                        help="Comma-separated ablation conditions.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    return parser.parse_args()


def load_probes(path: Optional[str], max_probes: Optional[int]) -> List[str]:
    if path is None:
        probes = list(DEFAULT_PROBES)
    else:
        probe_path = Path(path)
        if not probe_path.exists():
            raise FileNotFoundError(f"Probe file not found: {probe_path}")
        probes = [line.strip() for line in probe_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_probes is not None:
        probes = probes[:max_probes]
    if not probes:
        raise ValueError("No probes available.")
    return probes


def tokenize(text: str) -> Set[str]:
    return {
        token for token in re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
        if len(token) > 2 and token not in STOPWORDS
    }


def subject_from_query(query: str) -> Optional[str]:
    low = query.lower()
    patterns = [
        r"(.+?)\s+nedir",
        r"(.+?)\s+ne demek",
        r"(.+?)\s+ne işe",
        r"(.+?)\s+neyi",
        r"(.+?)\s+nasıl",
        r"(.+?)\s+neden",
    ]
    for pattern in patterns:
        match = re.search(pattern, low)
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(r"[^\wçğıöşüÇĞİÖŞÜ\s]", "", candidate).strip()
            if candidate:
                return candidate
    words = [w for w in tokenize(query)]
    return words[0] if words else None


def overlap_score(query: str, response: str, subject: Optional[str]) -> float:
    expected = tokenize(query)
    if subject:
        expected.add(subject.lower())
    if not expected:
        return 0.0
    actual = tokenize(response)
    return round(len(expected & actual) / len(expected), 4)


def copy_ratio(response: str, facts: Iterable[Dict[str, Any]]) -> float:
    response_low = (response or "").lower()
    if not response_low:
        return 0.0
    copied_chars = 0
    seen = set()
    for fact in facts:
        text = (fact.get("text") or "").strip()
        text_low = text.lower()
        if len(text_low) < 12 or text_low in seen:
            continue
        seen.add(text_low)
        if text_low in response_low:
            copied_chars += len(text_low)
    return round(min(1.0, copied_chars / max(1, len(response_low))), 4)


def source_counts(facts: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for fact in facts:
        source = fact.get("source") or "kb"
        counts[source] = counts.get(source, 0) + 1
    return counts


def response_changed(a: str, b: str) -> bool:
    return re.sub(r"\s+", " ", (a or "").strip()) != re.sub(r"\s+", " ", (b or "").strip())


def top_candidate_signature(row: Dict[str, Any]) -> str:
    candidates = row.get("top_candidates") or []
    if not candidates:
        return ""
    top = candidates[0]
    text = re.sub(r"\s+", " ", (top.get("text") or "").strip().lower())
    return f"{top.get('source')}::{text[:120]}"


def deterministic_probe_seed(probe: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(probe)) % (2 ** 32)


@contextmanager
def audit_limbic_awake(mergen: MergenBrain_v7):
    limbic = getattr(mergen, "limbic", None)
    if limbic is None:
        yield
        return

    was_running = getattr(limbic, "is_running", False)
    try:
        if not was_running:
            try:
                limbic.load_state()
            except Exception as exc:
                print(f"[Audit] Limbic state load failed: {exc}")
            limbic.is_running = True
        yield
    finally:
        if not was_running:
            limbic.is_running = False


@contextmanager
def isolated_runtime_state(mergen: MergenBrain_v7):
    conv = mergen.conv_memory
    analyzer = mergen.analyzer
    conv_state = {
        "turns": copy.deepcopy(conv.turns),
        "current_topics": copy.deepcopy(conv.current_topics),
        "topic_history": copy.deepcopy(conv.topic_history),
        "last_subject": conv.last_subject,
        "last_object": conv.last_object,
        "last_action": conv.last_action,
        "total_turns": conv.total_turns,
        "total_resolutions": conv.total_resolutions,
    }
    analyzer_state = {
        "memory": copy.deepcopy(analyzer.memory),
        "context_buffer": copy.deepcopy(analyzer.context_buffer),
        "last_subject": analyzer.last_subject,
        "last_intent": analyzer.last_intent,
        "total_analyses": analyzer.total_analyses,
        "low_confidence_count": analyzer.low_confidence_count,
    }
    original_conv_save = conv._save
    original_analyzer_save = analyzer._save_memory
    conv._save = lambda: None
    analyzer._save_memory = lambda: None
    try:
        yield
    finally:
        conv._save = original_conv_save
        analyzer._save_memory = original_analyzer_save
        conv.turns = conv_state["turns"]
        conv.current_topics = conv_state["current_topics"]
        conv.topic_history = conv_state["topic_history"]
        conv.last_subject = conv_state["last_subject"]
        conv.last_object = conv_state["last_object"]
        conv.last_action = conv_state["last_action"]
        conv.total_turns = conv_state["total_turns"]
        conv.total_resolutions = conv_state["total_resolutions"]
        analyzer.memory = analyzer_state["memory"]
        analyzer.context_buffer = analyzer_state["context_buffer"]
        analyzer.last_subject = analyzer_state["last_subject"]
        analyzer.last_intent = analyzer_state["last_intent"]
        analyzer.total_analyses = analyzer_state["total_analyses"]
        analyzer.low_confidence_count = analyzer_state["low_confidence_count"]


@contextmanager
def patched_recall(mergen: MergenBrain_v7):
    original = mergen._recall_knowledge
    bucket: Dict[str, Any] = {"facts": [], "elapsed_ms": 0.0, "metadata": {}}

    def wrapper(query: str, intent: str, subject: Optional[str], *args, **kwargs):
        start = time.perf_counter()
        facts = original(query, intent, subject, *args, **kwargs)
        bucket["elapsed_ms"] = (time.perf_counter() - start) * 1000.0
        bucket["facts"] = facts
        bucket["metadata"] = copy.deepcopy(getattr(mergen, "last_recall_metadata", {}))
        return facts

    mergen._recall_knowledge = wrapper
    try:
        yield bucket
    finally:
        mergen._recall_knowledge = original


@contextmanager
def condition_applied(mergen: MergenBrain_v7, condition: str):
    saved: Dict[str, Any] = {}
    condition = condition.strip()
    try:
        if condition == "no_rag":
            saved["rag"] = getattr(mergen, "rag", None)
            mergen.rag = None
        elif condition == "no_limbic":
            saved["limbic"] = getattr(mergen, "limbic", None)
            mergen.limbic = None
        elif condition == "no_semantic":
            saved["use_wernicke"] = getattr(mergen.enhanced_brain, "use_wernicke", False)
            mergen.enhanced_brain.use_wernicke = False
        elif condition == "no_hebbian_trace":
            trace = getattr(mergen.brain, "hebbian_trace", None)
            if trace is not None:
                saved["hebbian_trace"] = trace.clone()
                trace.zero_()
        elif condition == "baseline":
            pass
        else:
            raise ValueError(f"Unknown condition: {condition}")
        yield
    finally:
        if "rag" in saved:
            mergen.rag = saved["rag"]
        if "limbic" in saved:
            mergen.limbic = saved["limbic"]
        if "use_wernicke" in saved:
            mergen.enhanced_brain.use_wernicke = saved["use_wernicke"]
        if "hebbian_trace" in saved:
            mergen.brain.hebbian_trace.copy_(saved["hebbian_trace"])


def run_probe(mergen: MergenBrain_v7, probe: str, condition: str) -> Dict[str, Any]:
    with isolated_runtime_state(mergen):
        report = mergen.analyzer.analyze_intent(probe)
        subject = report.get("subject") or subject_from_query(probe)

        with condition_applied(mergen, condition):
            with patched_recall(mergen) as recall_info:
                wall_start = time.perf_counter()
                cpu_start = time.process_time()
                random.seed(deterministic_probe_seed(probe))
                response = mergen.respond(probe)
                wall_ms = (time.perf_counter() - wall_start) * 1000.0
                cpu_ms = (time.process_time() - cpu_start) * 1000.0

    facts = list(recall_info.get("facts") or [])
    recall_metadata = copy.deepcopy(recall_info.get("metadata") or {})
    return {
        "probe": probe,
        "condition": condition,
        "intent": report.get("primary_intent"),
        "subject": subject,
        "response": response,
        "wall_ms": round(wall_ms, 2),
        "cpu_ms": round(cpu_ms, 2),
        "recall_ms": round(float(recall_info.get("elapsed_ms") or 0.0), 2),
        "fact_count": len(facts),
        "source_counts": source_counts(facts),
        "rag_hit_count": sum(1 for f in facts if (f.get("source") or "") == "rag"),
        "semantic_fallback_used": bool(recall_metadata.get("semantic_fallback_used")),
        "top_candidates": recall_metadata.get("top_candidates", []),
        "avg_final_score": average_numeric(facts, "final_score"),
        "avg_hebbian_score": average_numeric(facts, "hebbian_score"),
        "avg_limbic_score": average_numeric(facts, "limbic_score"),
        "avg_rag_score": average_numeric(facts, "rag_score"),
        "limbic_fired": getattr(getattr(mergen, "limbic", None), "last_thought", ""),
        "topic_overlap": overlap_score(probe, response, subject),
        "copy_ratio": copy_ratio(response, facts),
        "response_length": len(response),
    }


def average_numeric(rows: Iterable[Dict[str, Any]], field: str) -> float:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(field, 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_condition: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        by_condition.setdefault(item["condition"], []).append(item)

    summary: Dict[str, Any] = {}
    for condition, rows in by_condition.items():
        count = max(1, len(rows))
        summary[condition] = {
            "count": len(rows),
            "avg_wall_ms": round(sum(r["wall_ms"] for r in rows) / count, 2),
            "avg_recall_ms": round(sum(r["recall_ms"] for r in rows) / count, 2),
            "avg_topic_overlap": round(sum(r["topic_overlap"] for r in rows) / count, 4),
            "avg_copy_ratio": round(sum(r["copy_ratio"] for r in rows) / count, 4),
            "avg_fact_count": round(sum(r["fact_count"] for r in rows) / count, 2),
            "avg_final_score": round(sum(r.get("avg_final_score", 0.0) for r in rows) / count, 4),
            "avg_hebbian_score": round(sum(r.get("avg_hebbian_score", 0.0) for r in rows) / count, 4),
            "avg_limbic_score": round(sum(r.get("avg_limbic_score", 0.0) for r in rows) / count, 4),
            "avg_rag_score": round(sum(r.get("avg_rag_score", 0.0) for r in rows) / count, 4),
            "semantic_fallback_count": sum(1 for r in rows if r.get("semantic_fallback_used")),
        }

    baseline = {r["probe"]: r for r in by_condition.get("baseline", [])}
    deltas: Dict[str, Any] = {}
    for condition, rows in by_condition.items():
        if condition == "baseline":
            continue
        comparable = [r for r in rows if r["probe"] in baseline]
        if not comparable:
            continue
        changed = sum(1 for r in comparable if response_changed(r["response"], baseline[r["probe"]]["response"]))
        ranking_changed = sum(
            1 for r in comparable
            if top_candidate_signature(r) != top_candidate_signature(baseline[r["probe"]])
        )
        score_changed = sum(
            1 for r in comparable
            if abs(float(r.get("avg_final_score", 0.0)) - float(baseline[r["probe"]].get("avg_final_score", 0.0))) > 0.01
        )
        deltas[condition] = {
            "changed_responses": changed,
            "changed_top_candidates": ranking_changed,
            "changed_scores": score_changed,
            "total": len(comparable),
            "changed_rate": round(changed / max(1, len(comparable)), 4),
            "ranking_changed_rate": round(ranking_changed / max(1, len(comparable)), 4),
            "score_changed_rate": round(score_changed / max(1, len(comparable)), 4),
        }

    return {"by_condition": summary, "deltas_vs_baseline": deltas}


def decision_gates(summary: Dict[str, Any]) -> List[str]:
    gates = []
    deltas = summary.get("deltas_vs_baseline", {})
    conditions = summary.get("by_condition", {})

    no_rag = deltas.get("no_rag", {}).get("changed_rate")
    no_limbic = deltas.get("no_limbic", {}).get("changed_rate")
    no_hebb = deltas.get("no_hebbian_trace", {}).get("changed_rate")
    no_hebb_score = deltas.get("no_hebbian_trace", {}).get("score_changed_rate")

    if no_rag is not None and no_limbic is not None:
        no_rag_effect = max(
            no_rag,
            deltas.get("no_rag", {}).get("ranking_changed_rate", 0.0),
            deltas.get("no_rag", {}).get("score_changed_rate", 0.0),
        )
        no_limbic_effect = max(
            no_limbic,
            deltas.get("no_limbic", {}).get("ranking_changed_rate", 0.0),
            deltas.get("no_limbic", {}).get("score_changed_rate", 0.0),
        )
        if no_rag_effect >= 0.5 and no_limbic_effect <= 0.2:
            gates.append("RAG has strong causal influence while Limbic appears weak in direct answers.")
        elif no_limbic_effect >= 0.5:
            gates.append("Limbic materially changes answers; inspect whether changes improve quality.")
        else:
            gates.append("No single ablation dominated answer changes; inspect per-probe details.")

    if no_hebb is not None:
        hebb_effect = max(no_hebb, no_hebb_score or 0.0, deltas.get("no_hebbian_trace", {}).get("ranking_changed_rate", 0.0))
        if hebb_effect <= 0.2:
            gates.append("Hebbian trace changes do not strongly affect current QA answers.")
        else:
            gates.append("Hebbian trace affects current QA answers; runtime coupling is measurable.")

    baseline = conditions.get("baseline", {})
    if baseline.get("avg_copy_ratio", 0) > 0.65:
        gates.append("Answers are mostly extractive recall rather than synthesized language.")
    if baseline.get("avg_wall_ms", 0) > 5000:
        gates.append("Baseline QA latency is high; profile RAG/Limbic activation and remaining fallbacks.")

    no_semantic_stats = conditions.get("no_semantic", {})
    if baseline and no_semantic_stats:
        gap = baseline.get("avg_wall_ms", 0) - no_semantic_stats.get("avg_wall_ms", 0)
        if gap > 5000:
            gates.append("Semantic path still creates a large latency gap; fallback policy needs inspection.")

    return gates


def build_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        f"# Mergen Cognitive Audit: {payload['run_name']}",
        "",
        f"- Created: `{payload['created_at']}`",
        f"- Probe count: `{len(payload['probes'])}`",
        f"- Conditions: `{', '.join(payload['conditions'])}`",
        f"- Knowledge facts: `{payload['system'].get('knowledge_facts')}`",
        f"- RAG records: `{payload['system'].get('rag_records')}`",
        "",
        "## Summary",
    ]
    for condition, stats in payload["summary"]["by_condition"].items():
        lines.append(
            f"- `{condition}`: wall `{stats['avg_wall_ms']}ms`, recall `{stats['avg_recall_ms']}ms`, "
            f"overlap `{stats['avg_topic_overlap']}`, copy `{stats['avg_copy_ratio']}`, facts `{stats['avg_fact_count']}`, "
            f"score `{stats.get('avg_final_score')}`, hebb `{stats.get('avg_hebbian_score')}`, "
            f"limbic `{stats.get('avg_limbic_score')}`, rag `{stats.get('avg_rag_score')}`, "
            f"semantic_fallbacks `{stats.get('semantic_fallback_count')}`"
        )
    lines.append("")
    lines.append("## Deltas")
    for condition, delta in payload["summary"].get("deltas_vs_baseline", {}).items():
        lines.append(
            f"- `{condition}`: response `{delta.get('changed_rate')}`, "
            f"top_candidate `{delta.get('ranking_changed_rate')}`, score `{delta.get('score_changed_rate')}`"
        )
    lines.append("")
    lines.append("## Decision Gates")
    for gate in payload["decision_gates"]:
        lines.append(f"- {gate}")
    if not payload["decision_gates"]:
        lines.append("- No strong decision gate triggered.")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    probes = load_probes(args.probes, args.max_probes)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("[Audit] Waking Mergen for cognitive audit...")
    mergen = MergenBrain_v7(verbose=False)

    results: List[Dict[str, Any]] = []
    dream_result: Optional[str] = None
    with audit_limbic_awake(mergen):
        for condition in conditions:
            if not args.quiet:
                print(f"[Audit] Condition: {condition}")
            for probe in probes:
                result = run_probe(mergen, probe, condition)
                results.append(result)
                if not args.quiet:
                    print(
                        f"  {probe} -> {result['wall_ms']}ms, "
                        f"facts={result['fact_count']}, overlap={result['topic_overlap']}"
                    )

        if args.dream_cycles > 0:
            if not 1 <= args.dream_cycles <= 1000:
                raise ValueError("--dream-cycles must be between 1 and 1000.")
            if not args.quiet:
                print(f"[Audit] Running Dream for {args.dream_cycles} cycles...")
            dream_result = mergen._run_dream_consolidation(cycles=args.dream_cycles)
            for probe in probes:
                results.append(run_probe(mergen, probe, "baseline_after_dream"))

    summary = summarize(results)
    payload = {
        "run_name": args.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "probes": probes,
        "conditions": conditions + (["baseline_after_dream"] if dream_result else []),
        "system": {
            "knowledge_facts": mergen.brain.knowledge_size(),
            "rag_records": mergen.rag.count() if getattr(mergen, "rag", None) else 0,
            "vocabulary": mergen.vocab.size(),
        },
        "dream_result": dream_result,
        "results": results,
        "summary": summary,
        "decision_gates": decision_gates(summary),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = logs_dir / f"{args.name}_{timestamp}.json"
    md_path = logs_dir / f"{args.name}_{timestamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")

    print(f"[Audit] Wrote {json_path}")
    print(f"[Audit] Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
