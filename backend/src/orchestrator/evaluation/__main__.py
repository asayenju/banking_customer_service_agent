"""Opt-in commands (they call the live model and need OPENAI_KEY in the repo-root .env).

    uv run python -m orchestrator.evaluation build [--count 50] [--set PATH] [--exclude PATH]
    uv run python -m orchestrator.evaluation label [--set PATH]
    uv run python -m orchestrator.evaluation run [--trials 3] [--temperature 0.0] [--set PATH]
    uv run python -m orchestrator.evaluation jev [--set PATH ...]   # needs JEV_KEY and the jev extra

The main set is evals/banking_eval_set.json. A holdout built with ``--exclude`` the
main set (the tasks it left out) checks that prompt changes were not tuned to it.
"""

import argparse
import asyncio
import json
from pathlib import Path

from orchestrator.evaluation.messages import build_cases, label_cases, load_cases, save_cases
from orchestrator.evaluation.runner import (
    DEFAULT_TRIALS,
    render,
    run_eval,
    summarize,
)
from orchestrator.evaluation.tasks import DATASET_PATH, DEFAULT_COUNT, load_tasks, select_tasks
from orchestrator.llm import LiteLLMClient
from orchestrator.llm_decomposer import LLMDecomposer


# Stronger than the decomposer's model, so the answer key is not graded by its own peer.
LABELING_MODEL = "openai/gpt-4o"


def main() -> None:
    parser = argparse.ArgumentParser(prog="orchestrator.evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="write the frozen set of customer messages")
    build.add_argument("--count", type=int, default=DEFAULT_COUNT)
    build.add_argument("--exclude", type=Path, default=None, help="skip tasks already in this set")
    label = sub.add_parser("label", help="relabel the saved messages with the labeling model")
    run = sub.add_parser("run", help="score the decomposer on the frozen set")
    jev = sub.add_parser("jev", help="how well Jev's confidence separates right intents from wrong ones")
    jev.add_argument("--sets", type=Path, nargs="+", default=[DATASET_PATH])
    for p in (build, label, run):
        p.add_argument("--set", type=Path, default=DATASET_PATH, dest="dataset")
    run.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--out", type=Path, default=None, help="also write per-trial JSON here")
    args = parser.parse_args()

    client = LiteLLMClient()
    if args.command in ("build", "label"):
        labeler = LiteLLMClient(LABELING_MODEL)
        if args.command == "build":
            skip = {c.task_id for c in load_cases(args.exclude)} if args.exclude else set()
            picked = select_tasks(load_tasks(), args.count, exclude=skip)
            cases = asyncio.run(build_cases(client, labeler, picked))
        else:
            cases = asyncio.run(label_cases(labeler, load_cases(args.dataset)))
        save_cases(cases, args.dataset)
        print(f"wrote {len(cases)} cases")
        return

    if args.command == "jev":
        from orchestrator.evaluation.jev_check import observe, render as render_jev
        from orchestrator.jev import make_jev_scorer

        every = [c for path in args.sets for c in load_cases(path)]
        observed = asyncio.run(observe(every, LLMDecomposer(client), make_jev_scorer()))
        print(render_jev(observed))
        return

    cases = load_cases(args.dataset)
    results = asyncio.run(
        run_eval(
            cases,
            lambda: LLMDecomposer(client, temperature=args.temperature),
            trials=args.trials,
        )
    )
    print(render(summarize(results, args.trials), results))
    if args.out:
        args.out.write_text(json.dumps([r.model_dump(mode="json") for r in results], indent=2))


if __name__ == "__main__":
    main()
