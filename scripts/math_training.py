"""Mergen V7 — Aritmetik egitim ve holdout degerlendirme scripti.

MathTeacher uzerinden problem uretir, MergenBrain_v7'nin learn_from_text API'si
ile ogretir, train/holdout ayrimli recall degerlendirmesi yapar.

Kullanim:
    python scripts/math_training.py --tier 0 --no-save
    python scripts/math_training.py --tier 1 --dream --dream-cycles 10
    python scripts/math_training.py --dry-run --tier 2
"""

import argparse
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.generators.math_teacher import MathTeacher, OP_SUBJECT, SAYI_ADI, OP_ADI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mergen V7 aritmetik egitim")
    p.add_argument("--tier", type=int, default=0, choices=[0, 1, 2, 3],
                   help="Zorluk kademesi: 0=toplama, 1=+cikarma, 2=+carpma, 3=+bolme")
    p.add_argument("--seed", type=int, default=42,
                   help="Train/holdout split seed (default: 42)")
    p.add_argument("--split", type=float, default=0.80,
                   help="Train orani 0.50-0.95 (default: 0.80)")
    p.add_argument("--reward", type=float, default=1.0,
                   help="Ogrenme reward degeri (default: 1.0)")
    p.add_argument("--dream", action="store_true",
                   help="Dream konsolidasyonu calistir")
    p.add_argument("--dream-cycles", type=int, default=10,
                   help="Dream cycle sayisi (default: 10)")
    p.add_argument("--epochs", type=int, default=5,
                   help="Her tier icin maksimum epoch sayisi (default: 5)")
    p.add_argument("--curriculum", action="store_true",
                   help="Mufredat tabanli sirali egitim (Tier 0 -> Tier 3)")
    p.add_argument("--no-early-stop", action="store_true",
                   help="Erken durdurmayi pasif yap")
    p.add_argument("--difficulty", type=int, default=0, choices=[0, 1, 2],
                   help="Zorluk derecesi: 0=0-9, 1=0-20, 2=0-99 (default: 0)")
    p.add_argument("--dry-run", action="store_true",
                   help="Brain yuklemeden plan onizlemesi goster")
    p.add_argument("--no-save", action="store_true",
                   help="Brain state kaydetme")
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> str | None:
    if not 0.50 <= args.split <= 0.95:
        return "--split must be between 0.50 and 0.95"
    if args.reward <= 0:
        return "--reward must be > 0"
    if not 1 <= args.dream_cycles <= 1000:
        return "--dream-cycles must be between 1 and 1000"
    if args.epochs <= 0:
        return "--epochs must be > 0"
    if args.difficulty not in [0, 1, 2]:
        return "--difficulty must be 0, 1, or 2"
    return None


def make_intent_report() -> Dict:
    return {
        "primary_intent": "EXPERIENCE",
        "confidence_score": 0.9,
        "sentiment": {"sentiment_score": 0.0, "excitement": 0.0},
        "subject": "aritmetik",
    }


def split_problems(problems: List[Dict], seed: int, train_ratio: float):
    rng = random.Random(seed)
    shuffled = list(problems)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * train_ratio)
    return shuffled[:cut], shuffled[cut:]


def evaluate(mergen, problems: List[Dict], top_k: int = 5, verbose: bool = False) -> Dict:
    """Recall degerlendirmesi. Her problem icin sorgu olustur, recall_raw ile ara."""
    per_op = defaultdict(lambda: {"total": 0, "hits": 0})
    total_hits = 0

    if verbose:
        print("\n  " + "-" * 75)
        print(f"  {'DURUM':<20s} | {'SORU':<35s} | {'MERGEN CEVABI'}")
        print("  " + "-" * 75)

    for prob in problems:
        a, b, op = prob['a'], prob['b'], prob['op']
        result = prob['result']
        op_adi = OP_ADI[op]
        a_adi = SAYI_ADI.get(a, str(a))
        b_adi = SAYI_ADI.get(b, str(b))
        r_adi = SAYI_ADI.get(result, str(result))

        query = f"{a_adi} {op_adi} {b_adi}"
        results = mergen.enhanced_brain.recall_raw(query, top_k=top_k)

        hit = False
        for r in results:
            text = r.get('text', '').lower()
            if r_adi in text and 'esittir' in text:
                hit = True
                break

        # Mergen'in verdigi en yuksek oncelikli cevabi cikarmaya calis
        actual_ans = "Yok"
        if results:
            top_hit = results[0].get('text', '').lower()
            parts = top_hit.split('esittir')
            if len(parts) > 1:
                ans_words = parts[1].strip().split()
                if ans_words:
                    # 'iki' ya da sayisal '2' olabilir
                    actual_ans = ans_words[0]

        if verbose:
            status = "[DOGRU]" if hit else f"[YANLIS (Beklenen: {r_adi})]"
            print(f"  {status:<20s} | {a} {op} {b} ({query}) -> {actual_ans}")

        op_name = OP_SUBJECT[op]
        per_op[op_name]["total"] += 1
        if hit:
            per_op[op_name]["hits"] += 1
            total_hits += 1

    if verbose:
        print("  " + "-" * 75 + "\n")

    total = len(problems)
    return {
        "total": total,
        "hits": total_hits,
        "rate": total_hits / max(1, total),
        "per_op": {k: {"total": v["total"],
                        "hits": v["hits"],
                        "rate": v["hits"] / max(1, v["total"])}
                   for k, v in per_op.items()},
    }


def print_dry_run(args, teacher, train_set, holdout_set):
    tier_names = {0: "toplama", 1: "toplama+cikarma",
                  2: "toplama+cikarma+carpma", 3: "tam (4 islem)"}
    print("=" * 55)
    print("  Mergen Aritmetik Egitim — DRY RUN")
    print("=" * 55)
    print(f"Tier         : {args.tier} ({tier_names.get(args.tier, '?')})")
    print(f"Seed         : {args.seed}")
    print(f"Split        : {args.split:.0%} train / {1-args.split:.0%} holdout")
    print(f"Train set    : {len(train_set)} problem")
    print(f"Holdout set  : {len(holdout_set)} problem")
    print(f"Reward       : {args.reward}")
    print(f"Dream        : {'evet (' + str(args.dream_cycles) + ' cycle)' if args.dream else 'hayir'}")
    print(f"Save         : {'hayir (--no-save)' if args.no_save else 'evet'}")
    print()

    per_op_train = defaultdict(int)
    per_op_holdout = defaultdict(int)
    for p in train_set:
        per_op_train[OP_SUBJECT[p['op']]] += 1
    for p in holdout_set:
        per_op_holdout[OP_SUBJECT[p['op']]] += 1

    print("Islem breakdown:")
    for op_name in ['toplama', 'cikarma', 'carpma', 'bolme']:
        t = per_op_train.get(op_name, 0)
        h = per_op_holdout.get(op_name, 0)
        if t + h > 0:
            print(f"  {op_name:10s} : train={t:4d}  holdout={h:3d}")
    print()

    print("Ornek train fact'lar (ilk 5):")
    for p in train_set[:5]:
        print(f"  {MathTeacher.format_fact(p)}")
    print()
    print("Ornek holdout fact'lar (ilk 5):")
    for p in holdout_set[:5]:
        print(f"  {MathTeacher.format_fact(p)}")


def print_report(args, train_set, holdout_set, facts_added,
                 avg_words, kb_before, kb_after, train_eval, holdout_eval,
                 trace_concepts, elapsed):
    tier_names = {0: "toplama", 1: "toplama+cikarma",
                  2: "toplama+cikarma+carpma", 3: "tam (4 islem)"}
    print()
    print("=" * 55)
    print("  Mergen Aritmetik Egitim Raporu")
    print("=" * 55)
    print(f"Tier         : {args.tier} ({tier_names.get(args.tier, '?')})")
    print(f"Seed         : {args.seed}")
    print(f"Train/Holdout: {len(train_set)} / {len(holdout_set)}")
    print(f"Sure         : {elapsed:.1f}s")
    print()

    print("--- Egitim ---")
    print(f"facts_added       : {facts_added}")
    print(f"avg_words_learned : {avg_words:.1f}")
    print(f"kb_size_before    : {kb_before}")
    print(f"kb_size_after     : {kb_after}")
    print()

    print("--- Degerlendirme ---")
    header = f"{'':15s} {'Train':>8s} {'Holdout':>8s}"
    print(header)
    all_ops = sorted(set(list(train_eval['per_op'].keys()) +
                         list(holdout_eval['per_op'].keys())))
    for op_name in all_ops:
        t_rate = train_eval['per_op'].get(op_name, {}).get('rate', 0.0)
        h_rate = holdout_eval['per_op'].get(op_name, {}).get('rate', 0.0)
        print(f"  {op_name:13s} {t_rate:8.2f} {h_rate:8.2f}")
    print(f"  {'TOPLAM':13s} {train_eval['rate']:8.2f} {holdout_eval['rate']:8.2f}")
    gap = train_eval['rate'] - holdout_eval['rate']
    print(f"\ngeneralization_gap : {gap:.2f}")
    print()

    print("--- Hebbian ---")
    print(f"trace_concepts_reinforced : {trace_concepts} unique")
    print()

    train_ok = train_eval['rate'] >= 0.70
    holdout_ok = holdout_eval['rate'] >= 0.40
    verdict = "PASS" if (train_ok and holdout_ok) else "FAIL"
    print(f"--- Karar ---")
    print(f"recall_pass (train>=0.70 & holdout>=0.40): {verdict}")
    if not train_ok:
        print(f"  ! train_recall_rate {train_eval['rate']:.2f} < 0.70")
    if not holdout_ok:
        print(f"  ! holdout_recall_rate {holdout_eval['rate']:.2f} < 0.40")


def main() -> int:
    args = parse_args()
    err = validate_args(args)
    if err:
        print(f"[MathTraining] Argument error: {err}")
        return 2

    # --- Brain yukle ---
    from brain import MergenBrain_v7 as DigitalBrain

    print("[MathTraining] Mergen yukleniyor...")
    try:
        mergen = DigitalBrain(verbose=False)
    except Exception as exc:
        print(f"[MathTraining] Error initializing Mergen: {exc}")
        return 1

    # Deterministic intent report for learning
    intent_report = make_intent_report()

    # Determine which tiers to train
    if args.curriculum:
        tiers_to_train = [0, 1, 2, 3]
        print("\n" + "=" * 60)
        print("  MUFREDAT TABANLI ARDISIK EGITIM BASLIYOR (Tier 0 -> Tier 3)")
        print("=" * 60)
    else:
        tiers_to_train = [args.tier]

    for tier in tiers_to_train:
        tier_names = {0: "Toplama", 1: "Çıkarma", 2: "Çarpma", 3: "Bölme"}
        print("\n" + "=" * 60)
        print(f"  Tier {tier} Eğitimi: {tier_names[tier]}")
        print("=" * 60)

        teacher = MathTeacher(tier=tier, difficulty=args.difficulty)
        all_problems = teacher.enumerate_all()
        if not all_problems:
            print(f"[MathTraining] Tier {tier} icin problem uretilemedi.")
            return 1

        train_set, holdout_set = split_problems(all_problems, args.seed, args.split)

        if args.dry_run:
            args.tier = tier
            print_dry_run(args, teacher, train_set, holdout_set)
            continue

        kb_before = len(mergen.brain.knowledge_base)
        t0 = time.time()
        import re as _re

        passed = False
        train_eval = {}
        holdout_eval = {}
        trace_concept_set = set()
        total_words = 0
        facts_added_tier = 0

        # Epoch loop
        for epoch in range(1, args.epochs + 1):
            epoch_train = list(train_set)
            random.shuffle(epoch_train)

            for prob in epoch_train:
                fact_text = MathTeacher.format_fact(prob)

                # Check duplication manually in KB
                exists = False
                for existing in mergen.brain.knowledge_base:
                    if existing['text'] == fact_text:
                        exists = True
                        break

                if not exists:
                    kb_idx = len(mergen.brain.knowledge_base)
                    tokens = _re.findall(r'\w+', fact_text.lower())
                    concept_ids = []
                    for tok in tokens:
                        if mergen.vocab.contains(tok):
                            cid = mergen.vocab.get_id(tok)
                            if cid not in concept_ids:
                                concept_ids.append(cid)

                    mergen.brain.knowledge_base.append({
                        'text': fact_text,
                        'concept_ids': concept_ids,
                        'weight': args.reward,
                        'access_count': 0,
                    })
                    for cid in concept_ids:
                        if cid not in mergen.brain.concept_index:
                            mergen.brain.concept_index[cid] = []
                        mergen.brain.concept_index[cid].append(kb_idx)
                    facts_added_tier += 1

                result = mergen.brain.learn_from_text(
                    text=fact_text,
                    vocab=mergen.vocab,
                    intent_report=intent_report,
                    learning_rate=0.02,
                    reward=args.reward,
                    store_in_kb=False,
                )
                wl = result.get('words_learned', 0)
                total_words += wl
                for w in result.get('matched_words', []):
                    trace_concept_set.add(w)

            # Optional dream consolidation after epoch
            if args.dream:
                dream = getattr(mergen, 'dream', None)
                if dream and hasattr(dream, 'run_dream_cycle'):
                    for c in range(1, args.dream_cycles + 1):
                        dream.run_dream_cycle()

            # Evaluate at the end of epoch
            train_eval = evaluate(mergen, train_set, verbose=False)
            holdout_eval = evaluate(mergen, holdout_set, verbose=False)

            train_rate = train_eval['rate']
            holdout_rate = holdout_eval['rate']

            print(f"  Epoch {epoch}/{args.epochs} | Train Recall: {train_rate:.2%} | Holdout Recall: {holdout_rate:.2%}")

            # Early stopping check
            if not args.no_early_stop and train_rate >= 0.90 and holdout_rate >= 0.60:
                print(f"  [MathTraining] Hedef basariya ulasildi, erken durduruluyor. (Train: {train_rate:.2%}, Holdout: {holdout_rate:.2%})")
                passed = True
                break
        else:
            # Check final epoch
            train_rate = train_eval.get('rate', 0.0)
            holdout_rate = holdout_eval.get('rate', 0.0)
            if train_rate >= 0.90 and holdout_rate >= 0.60:
                passed = True

        elapsed = time.time() - t0
        kb_after = len(mergen.brain.knowledge_base)
        avg_words = total_words / max(1, len(train_set) * args.epochs) # approximate

        # Print Verbose Detailed Output for Holdout Set on Tier Completion
        print(f"\n[MathTraining] Tier {tier} Detayli Holdout Degerlendirmesi:")
        holdout_eval = evaluate(mergen, holdout_set, verbose=True)

        args.tier = tier
        print_report(args, train_set, holdout_set, facts_added_tier,
                     avg_words, kb_before, kb_after, train_eval, holdout_eval,
                     len(trace_concept_set), elapsed)

        # STRICT HARD STOP POLICY
        if not passed:
            raise RuntimeError(
                f"[CURRICULUM HALT] Tier {tier} egitiminde basarisiz olundu (Hedef: Train>=90%, Holdout>=60%). "
                f"Agirlik matrisinin zehirlenmesini onlemek icin egitim DURDURULDU. "
                f"Son Skorlar -> Train: {train_eval.get('rate', 0.0):.2%}, Holdout: {holdout_eval.get('rate', 0.0):.2%}"
            )

        print(f"[MathTraining] ✓ Tier {tier} egitimi basariyla tamamlandi ve gecildi.")

        # Save brain state after each successful tier
        if not args.no_save:
            print("[MathTraining] Brain state kaydediliyor...")
            try:
                save_result = mergen.save()
                print(f"[MathTraining] Agirliklar kaydedildi: {save_result}")
            except Exception as e:
                print(f"[MathTraining] Kaydetme hatasi: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
