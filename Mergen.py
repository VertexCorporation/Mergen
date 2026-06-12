"""
╔══════════════════════════════════════════════════════════════════════╗
║          MERGEN — DIGITAL BRAIN v7.0 (Main Entry Point)              ║
║                                                                      ║
║  "A thinking digital brain — not a chatbot."                        ║
║                                                                      ║
║  ARCHITECTURE:                                                       ║
║    Wernicke (perception) → Intent Analysis → Brain Processing       ║
║    → Knowledge Recall → Response Synthesis → Broca (expression)     ║
║                                                                      ║
║  Author:  Vertex Corporation — Mergen Project                       ║
║  License: Apache-2.0                                                ║
╚══════════════════════════════════════════════════════════════════════╝

USAGE:
    python Mergen.py

COMMANDS:
    /exit, /quit, /çık  → shutdown (saves state)
    /stats               → show telemetry
    /introspect          → Mergen's self-model
    /clear               → clear conversation memory
    /help                → show commands
    oku:path/to/file.txt → read and learn from file
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the v7.0 brain
from brain import MergenBrain_v7


def main():
    """Entry point — launches the Digital Brain v7.0."""
    mergen = MergenBrain_v7(verbose=True)
    mergen.run()


if __name__ == "__main__":
    main()
