"""Calibrate SCORE_THRESHOLD.

FAISS returns L2 distance (lower = more similar). For each threshold we show:
  * in-scope kept   - JEE questions whose best chunk would be kept (want HIGH)
  * out-of-scope dropped - unrelated questions that would be rejected (want HIGH)
Pick the threshold that balances both, then set SCORE_THRESHOLD in your environment.
"""
import json

from app.rag import JEEAgent


QUESTIONS_FILE = "evaluation/questions.json"


def main():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        questions = json.load(file)

    agent = JEEAgent()

    in_scores = []
    out_scores = []

    print("\n======================================")
    print("RETRIEVAL THRESHOLD ANALYSIS")
    print("======================================\n")

    for index, item in enumerate(questions, start=1):

        question = item["question"]

        results = agent.vectorstore.similarity_search_with_score(
            question,
            k=5
        )

        if not results:
            continue

        best_score = min(float(score) for _, score in results)

        is_out = item.get("out_of_scope", False)
        (out_scores if is_out else in_scores).append(best_score)

        tag = "OUT" if is_out else "IN "
        print(
            f"{index:02d}. [{tag}] "
            f"Best score = {best_score:.4f} | "
            f"{question}"
        )

    if not in_scores:
        print("No scores collected - is the vector store empty?")
        return

    lo = int(min(in_scores + out_scores))
    hi = int(max(in_scores + out_scores)) + 5
    step = max((hi - lo) // 10, 1)
    thresholds = list(range(lo, hi + 1, step))

    print("\n======================================")
    print("THRESHOLD COMPARISON")
    print("======================================\n")

    for threshold in thresholds:

        kept = sum(s <= threshold for s in in_scores)
        dropped = sum(s > threshold for s in out_scores)

        line = (
            f"Threshold {threshold:>3}: "
            f"in-scope kept {kept}/{len(in_scores)} "
            f"({kept / len(in_scores) * 100:5.1f}%)"
        )
        if out_scores:
            line += (
                f" | out-of-scope dropped {dropped}/{len(out_scores)} "
                f"({dropped / len(out_scores) * 100:5.1f}%)"
            )
        print(line)

    print("\n======================================")
    print("SCORE STATISTICS (in-scope)")
    print("======================================")
    print(f"Minimum: {min(in_scores):.4f}")
    print(f"Maximum: {max(in_scores):.4f}")
    print(f"Average: {sum(in_scores) / len(in_scores):.4f}")
    if out_scores:
        print(f"Out-of-scope minimum: {min(out_scores):.4f}")


if __name__ == "__main__":
    main()
