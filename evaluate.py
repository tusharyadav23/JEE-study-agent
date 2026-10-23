"""Evaluate retrieval quality on evaluation/questions.json.

Reports three things:
  * coverage   - in-scope questions that returned at least one chunk
  * correctness - in-scope questions where a retrieved chunk contains an expected keyword
  * rejection  - out-of-scope questions that were correctly rejected (no chunks returned)
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

    in_scope = [q for q in questions if not q.get("out_of_scope")]
    out_scope = [q for q in questions if q.get("out_of_scope")]

    covered = 0
    correct = 0
    rejected = 0

    print("\n==============================")
    print("JEE AGENT EVALUATION")
    print("==============================\n")

    for index, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]
        results = agent.retrieve(question)

        if item.get("out_of_scope"):
            passed = len(results) == 0
            rejected += passed
            kind = "OUT-OF-SCOPE"
        else:
            has_results = len(results) > 0
            covered += has_results

            text = " ".join(
                doc.page_content for doc in results
            ).lower()

            keywords = item.get("expected_keywords", [])
            passed = has_results and (
                not keywords
                or any(k.lower() in text for k in keywords)
            )
            correct += passed
            kind = "IN-SCOPE"

        status = "PASS" if passed else "FAIL"

        print(
            f"{index:02d}. "
            f"[{status}] "
            f"({kind}) "
            f"{question}"
        )

    print("\n==============================")
    print(
        f"Coverage     : {covered}/{len(in_scope)} "
        f"({covered / max(len(in_scope), 1) * 100:.2f}%)"
    )
    print(
        f"Correctness  : {correct}/{len(in_scope)} "
        f"({correct / max(len(in_scope), 1) * 100:.2f}%)"
    )
    if out_scope:
        print(
            f"Rejection    : {rejected}/{len(out_scope)} "
            f"({rejected / len(out_scope) * 100:.2f}%)"
        )
    print("==============================")


if __name__ == "__main__":
    main()
