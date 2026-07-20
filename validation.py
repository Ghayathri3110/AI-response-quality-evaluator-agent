import random
from datasets import load_dataset

from relevance_agent import judge_relevance
from accuracy_agent import judge_accuracy
from hallucination_agent import detect_hallucination
from retrieval_agent import retrieve_context


MIN_REASONING_WORDS = 6
NUM_TEST_PAIRS = 8



# Build benchmark test cases from SQuAD

def build_squad_cases(n: int = NUM_TEST_PAIRS, seed: int = 42):
    """
    Pulls from the same 'train' split used in knowledge_builder.py, so
    retrieval has a real chance of surfacing matching indexed chunks.
    SQuAD doesn't ship 'wrong' answers, so an obviously false claim is
    injected into the correct answer to create a reliable wrong case.
    """
    random.seed(seed)
    squad = load_dataset("rajpurkar/squad", split="train[:2000]")
    sample = random.sample(list(squad), n)

    cases = []
    for row in sample:
        correct_answer = row["answers"]["text"][0]
        wrong_answer = (
            f"{correct_answer} This fact was actually first discovered by a "
            f"celebrity using a time machine in the year 3000."
        )
        cases.append({
            "question": row["question"],
            "correct_answer": correct_answer,
            "wrong_answer": wrong_answer,
            "reference_answer": correct_answer,
        })
    return cases

#single tests

def check_reasoning_quality(text: str) -> bool:
    return bool(text) and len(text.split()) >= MIN_REASONING_WORDS


def evaluate_case(case: dict, label: str, answer: str) -> dict:
    question = case["question"]
    reference_answer = case["reference_answer"]

    contexts = retrieve_context(question, top_k=2)

    relevance = judge_relevance(question, answer)
    accuracy = judge_accuracy(answer, reference_answer, contexts)
    hallucination = detect_hallucination(answer, contexts)

    reasoning_ok = (
        check_reasoning_quality(relevance["reasoning"])
        and check_reasoning_quality(accuracy["evidence"])
        and check_reasoning_quality(hallucination["reasoning"])
    )

    return {
        "question": question,
        "label": label,
        "answer": answer,
        "relevance_score": relevance["relevance_score"],
        "accuracy_score": accuracy["accuracy_score"],
        "hallucination_risk": hallucination["hallucination_risk"],
        "flagged_count": len(hallucination["flagged_statements"]),
        "reasoning_ok": reasoning_ok,
    }


def main():
    print("Building benchmark test cases from SQuAD...")
    cases = build_squad_cases(NUM_TEST_PAIRS)
    print(f"Loaded {len(cases)} benchmark pairs -> {len(cases) * 2} total test cases (correct + wrong each)\n")

    results = []
    for i, case in enumerate(cases):
        print(f"Evaluating pair {i + 1}/{len(cases)}...")
        results.append(evaluate_case(case, "correct", case["correct_answer"]))
        results.append(evaluate_case(case, "wrong", case["wrong_answer"]))

    #Report
    print("\n" + "=" * 88)
    print(f"{'LABEL':<10}{'ACCURACY':<10}{'RELEVANCE':<11}{'HALLUC RISK':<13}{'FLAGS':<7}{'REASONING'}")
    print("=" * 88)

    correct_acc, wrong_acc = [], []
    reasoning_ok_count = 0

    for r in results:
        print(
            f"{r['label']:<10}{r['accuracy_score']:<10}"
            f"{r['relevance_score']:<11}{r['hallucination_risk']:<13}"
            f"{r['flagged_count']:<7}{'OK' if r['reasoning_ok'] else 'WEAK'}"
        )
        if r["reasoning_ok"]:
            reasoning_ok_count += 1
        (correct_acc if r["label"] == "correct" else wrong_acc).append(r["accuracy_score"])

    avg_correct = sum(correct_acc) / len(correct_acc) if correct_acc else 0
    avg_wrong = sum(wrong_acc) / len(wrong_acc) if wrong_acc else 0

    print("=" * 88)
    print(f"Average accuracy score — correct answers : {avg_correct:.2f}")
    print(f"Average accuracy score — wrong answers   : {avg_wrong:.2f}")
    print(f"Scoring consistency: {'PASS' if avg_correct > avg_wrong else 'FAIL'} "
          f"(correct answers should score higher than wrong answers)")
    print(f"Reasoning quality: {reasoning_ok_count}/{len(results)} responses had substantive reasoning "
          f"(>= {MIN_REASONING_WORDS} words)")
    print("=" * 88)


if __name__ == "__main__":
    main()