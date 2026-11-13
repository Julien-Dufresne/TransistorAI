
"""Run TransistorAI translation mistake regression tests."""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from evaluation.hybrid_error_analysis import analyze_hybrid
from evaluation.regression_cases import CASES


OUTPUT_DIR = Path("results/regression_tests")


def normalize(value):
    return str(value).strip().casefold()


def evaluate_case(case):
    (
        case_id,
        group,
        source,
        reference,
        prediction,
        expected,
    ) = case

    try:
        analysis = analyze_hybrid(
            source=source,
            reference=reference,
            prediction=prediction,
        )

        mistakes = analysis.get("mistakes", [])
        feedback = analysis.get("general_feedback", [])
        diagnostics = analysis.get("diagnostics", [])

        detected = [m["category"] for m in mistakes]

        # Check for exact duplicate corrections,
        # independently of the category label.
        correction_pairs = [
            (
                normalize(m["original_text"]),
                normalize(m["corrected_text"]),
            )
            for m in mistakes
        ]

        duplicates = len(correction_pairs) - len(
            set(correction_pairs)
        )

        expected_set = set(expected)
        detected_set = set(detected)

        missing = sorted(expected_set - detected_set)
        unexpected = sorted(detected_set - expected_set)

        passed = (
            not missing
            and not unexpected
            and duplicates == 0
        )

        # Feedback is tracked separately because some
        # uncertain suggestions are not confirmed mistakes.
        feedback_on_correct = (
            not expected and bool(feedback)
        )

        return {
            "id": case_id,
            "group": group,
            "source": source,
            "reference": reference,
            "prediction": prediction,
            "expected": expected,
            "detected": detected,
            "passed": passed,
            "missing": missing,
            "unexpected": unexpected,
            "duplicate_count": duplicates,
            "feedback_on_correct": feedback_on_correct,
            "mistakes": mistakes,
            "feedback": feedback,
            "diagnostics": diagnostics,
            "warning": analysis.get("warning"),
            "error": None,
        }

    except Exception as exc:
        return {
            "id": case_id,
            "group": group,
            "source": source,
            "reference": reference,
            "prediction": prediction,
            "expected": expected,
            "detected": [],
            "passed": False,
            "missing": expected,
            "unexpected": [],
            "duplicate_count": 0,
            "feedback_on_correct": False,
            "mistakes": [],
            "feedback": [],
            "diagnostics": [],
            "warning": None,
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    cases = CASES[:args.limit]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] "
            f"{case[0]} — {case[1]}",
            flush=True,
        )

        result = evaluate_case(case)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"  {status} | "
            f"expected={result['expected']} | "
            f"detected={result['detected']} | "
            f"feedback={len(result['feedback'])}"
        )

        if result["error"]:
            print("  ERROR:", result["error"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = OUTPUT_DIR / f"regression_{timestamp}.json"
    csv_path = OUTPUT_DIR / f"regression_{timestamp}.csv"

    json_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    fields = [
        "id", "group", "passed", "expected", "detected",
        "missing", "unexpected", "duplicate_count",
        "feedback_on_correct", "feedback_count",
        "warning", "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for result in results:
            writer.writerow({
                field: (
                    len(result["feedback"])
                    if field == "feedback_count"
                    else json.dumps(result[field], ensure_ascii=False)
                    if field in {
                        "expected", "detected",
                        "missing", "unexpected",
                    }
                    else result.get(field)
                )
                for field in fields
            })

    total = len(results)
    passed = sum(r["passed"] for r in results)

    groups = sorted({r["group"] for r in results})

    print("\n" + "=" * 60)
    print("TRANSISTORAI REGRESSION REPORT")
    print("=" * 60)
    print(f"Passed: {passed}/{total} ({passed / total:.1%})")
    print(f"Failed: {total - passed}")
    print(
        "Duplicate corrections:",
        sum(r["duplicate_count"] for r in results),
    )
    print(
        "Correct cases with feedback:",
        sum(r["feedback_on_correct"] for r in results),
    )

    print("\nRESULTS BY GROUP")
    for group in groups:
        subset = [r for r in results if r["group"] == group]
        group_passed = sum(r["passed"] for r in subset)
        print(f"  {group}: {group_passed}/{len(subset)}")

    print("\nFAILED CASES")
    for result in results:
        if result["passed"]:
            continue

        print(f"  {result['id']}: {result['group']}")
        print(f"    Missing: {result['missing']}")
        print(f"    Unexpected: {result['unexpected']}")
        print(f"    Duplicates: {result['duplicate_count']}")

    print("\nReports:")
    print(" ", json_path)
    print(" ", csv_path)


if __name__ == "__main__":
    main()
