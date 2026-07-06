import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


SUMMARY_PATTERNS = {
    "total_experiences": r"Total experiences:\s+(\d+)",
    "processed_experiences": r"Processed experiences:\s+(\d+)",
    "failed_experiences": r"Failed experiences:\s+(\d+)",
    "words_learned": r"Words learned:\s+(\d+)",
    "kb_facts": r"KB facts:\s+(\d+)",
    "rag_indexed": r"RAG indexed:\s+(\d+)",
    "hebbian_bridge_updates": r"Hebbian bridge updates:\s+(\d+)",
    "brain_saved": r"Brain saved:\s+(True|False)",
    "limbic_saved": r"Limbic \.mx saved:\s+(True|False)",
    "last_fired_concepts": r"Last fired concepts:\s+(.+)",
    "dream": r"Dream:\s+([a-zA-Z]+)(?:\s+\((\d+)\s+cycles\))?",
}


EVAL_PATTERNS = {
    "evaluation_status": r"Status:\s+([a-zA-Z]+)",
    "kb_probes": r"KB probes:\s+(\d+)/(\d+)\s+\((\d+)%\)",
    "rag_probes": r"RAG probes:\s+(\d+)/(\d+)\s+\((\d+)%\)|RAG probes:\s+skipped",
    "limbic_overlap": r"Limbic overlap:\s+(\d+)/(\d+)\s+\((\d+)%\)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive Mergen playground metrics to logs/")
    parser.add_argument("--input", required=True, help="Path to a simulation_playground transcript.")
    parser.add_argument("--name", default=None, help="Optional run name for output files.")
    parser.add_argument("--logs-dir", default="logs", help="Directory for archived metrics (default: logs).")
    return parser.parse_args()


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input transcript not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    return value == "True"


def parse_summary(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for key, pattern in SUMMARY_PATTERNS.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if not match:
            continue
        if key == "dream":
            status = match.group(1)
            cycles = match.group(2)
            data[key] = {
                "status": status,
                "cycles": int(cycles) if cycles is not None else None,
            }
        elif key in {"brain_saved", "limbic_saved"}:
            data[key] = _bool(match.group(1))
        elif key == "last_fired_concepts":
            data[key] = match.group(1).strip()
        else:
            data[key] = int(match.group(1))

    eval_block = re.search(
        r"Evaluation:\s*\n\s+Status:\s+([a-zA-Z]+)\s*\n\s+KB probes:\s+(\d+)/(\d+)\s+\((\d+)%\)\s*\n\s+RAG probes:\s+(.+?)\s*\n\s+Limbic overlap:\s+(\d+)/(\d+)\s+\((\d+)%\)",
        text,
        flags=re.MULTILINE,
    )
    if eval_block:
        rag_line = eval_block.group(5).strip()
        rag_match = re.match(r"(\d+)/(\d+)\s+\((\d+)%\)", rag_line)
        data["evaluation"] = {
            "status": eval_block.group(1),
            "kb_passes": int(eval_block.group(2)),
            "kb_total": int(eval_block.group(3)),
            "kb_rate": int(eval_block.group(4)),
            "rag": None if rag_line == "skipped" else {
                "passes": int(rag_match.group(1)) if rag_match else None,
                "total": int(rag_match.group(2)) if rag_match else None,
                "rate": int(rag_match.group(3)) if rag_match else None,
            },
            "limbic_passes": int(eval_block.group(6)),
            "limbic_total": int(eval_block.group(7)),
            "limbic_rate": int(eval_block.group(8)),
        }
    elif "Evaluation:           skipped" in text or "Evaluation: skipped" in text:
        data["evaluation"] = {"status": "skipped"}

    return data


def build_markdown(run_name: str, transcript_name: str, metrics: Dict[str, Any]) -> str:
    dream = metrics.get("dream", {})
    evaluation = metrics.get("evaluation", {})
    lines = [
        f"# Playground Metrics: {run_name}",
        "",
        f"- Transcript: `{transcript_name}`",
        f"- Archived at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Total experiences: `{metrics.get('total_experiences', '-')}`",
        f"- Processed experiences: `{metrics.get('processed_experiences', '-')}`",
        f"- Failed experiences: `{metrics.get('failed_experiences', '-')}`",
        f"- Words learned: `{metrics.get('words_learned', '-')}`",
        f"- KB facts: `{metrics.get('kb_facts', '-')}`",
        f"- RAG indexed: `{metrics.get('rag_indexed', '-')}`",
        f"- Hebbian bridge updates: `{metrics.get('hebbian_bridge_updates', '-')}`",
        f"- Last fired concepts: `{metrics.get('last_fired_concepts', '-')}`",
        f"- Brain saved: `{metrics.get('brain_saved', '-')}`",
        f"- Limbic saved: `{metrics.get('limbic_saved', '-')}`",
    ]
    if dream:
        lines.append(f"- Dream: `{dream.get('status', '-')}`")
        if dream.get("cycles") is not None:
            lines.append(f"- Dream cycles: `{dream.get('cycles')}`")
    if evaluation:
        lines.append(f"- Evaluation status: `{evaluation.get('status', '-')}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    transcript = read_text(input_path)
    metrics = parse_summary(transcript)

    run_name = args.name or input_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = logs_dir / f"{run_name}_{timestamp}.json"
    md_path = logs_dir / f"{run_name}_{timestamp}.md"

    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(run_name, input_path.name, metrics), encoding="utf-8")

    print(f"[Archive] Wrote {json_path}")
    print(f"[Archive] Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
