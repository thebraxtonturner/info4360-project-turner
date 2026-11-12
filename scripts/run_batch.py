"""Send the Claude labeling requests through the Batch API. Run from the repo root, by hand.

Steps (see README, "Final labeling run"):
    python scripts/run_batch.py prepare                 # NER-clean the text once (no API)
    python scripts/run_batch.py dry-run --scope test    # build requests + cost estimate (no API)
    python scripts/run_batch.py submit  --scope test    # 100 reviews, asks you to type yes
    python scripts/run_batch.py status  --name test
    python scripts/run_batch.py fetch   --name test     # saves results to data/labels_final.jsonl
    python scripts/run_batch.py dry-run --scope full
    python scripts/run_batch.py submit  --scope full    # everything not labeled yet
    python scripts/run_batch.py status  --name full
    python scripts/run_batch.py fetch   --name full

Safety:
- A request whose result is already saved is never built again, so the full run skips the test reviews
  and a second submit only sends what failed.
- A batch name can be submitted once (data/batch/<name>_state.json). Use a new --name to retry failures.
- The API key is read from .env and never printed.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_reviews
from src import batch_labels as bl


def requests_for(scope, df, clean):
    done = bl.done_ids()
    if scope == "test":
        return bl.build_requests(df, clean, bl.test_sample_ids(df), done=done)
    return bl.build_requests(df, clean, repeats=bl.consistency_ids(df), done=done)


def cmd_prepare(args, df):
    if bl.CLEAN_PATH.exists() and not args.force:
        print(f"{bl.CLEAN_PATH.name} already exists (use --force to rebuild)")
        return
    out = bl.build_clean_text(df)
    print(f"Saved {len(out):,} reviews to {bl.CLEAN_PATH}")


def cmd_dry_run(args, df):
    clean = bl.load_clean_text()
    reqs = requests_for(args.scope, df, clean)
    bl.BATCH_DIR.mkdir(exist_ok=True)
    path = bl.BATCH_DIR / f"{args.scope}_requests_preview.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in reqs:
            f.write(json.dumps(r) + "\n")
    est = bl.estimate_cost(reqs)
    boxes = {b: sum(r["custom_id"].startswith(b + "-") for r in reqs) for b in bl.BOXES}
    repeats = sum(r["custom_id"].endswith("-r2") for r in reqs)
    print(f"Scope '{args.scope}': {est['requests']:,} requests to send "
          f"({boxes['neg']:,} negative boxes incl. {repeats} repeats, {boxes['pos']:,} positive boxes); "
          f"{len(bl.done_ids()):,} already labeled and skipped")
    print(f"Estimated cost with the Batch discount: about ${est['cost_usd_est']:.2f} "
          f"(~{est['input_tokens_est']:,} input + ~{est['output_tokens_est']:,} output tokens)")
    if args.command == "dry-run":
        print(f"Requests written to {path} so you can look at them first. Nothing was sent.")
    return reqs


def cmd_submit(args, df):
    name = args.name or args.scope
    state_path = bl.BATCH_DIR / f"{name}_state.json"
    if state_path.exists():
        sys.exit(f"Batch '{name}' was already submitted ({state_path}). Use status/fetch, "
                 f"or a new --name to send only what is still missing.")
    reqs = cmd_dry_run(args, df)
    if not reqs:
        sys.exit("Nothing to send: every box in this scope already has a label.")
    if not args.yes and input(f"\nSend {len(reqs):,} requests to the API as batch '{name}'? Type yes: ").strip() != "yes":
        sys.exit("Not sent.")

    from src.complaints import get_client
    client = get_client()
    batch = client.messages.batches.create(requests=reqs)
    state = {"name": name, "scope": args.scope, "batch_id": batch.id, "requests": len(reqs),
             "created_at": str(batch.created_at)}
    state_path.write_text(json.dumps(state, indent=2))
    print(f"Submitted batch {batch.id} ({len(reqs):,} requests). Saved to {state_path}.")
    print(f"Check it with: python scripts/run_batch.py status --name {name}")


def load_state(name):
    path = bl.BATCH_DIR / f"{name}_state.json"
    if not path.exists():
        sys.exit(f"No batch named '{name}' ({path} not found).")
    return json.loads(path.read_text())


def cmd_status(args, df):
    from src.complaints import get_client
    state = load_state(args.name)
    b = get_client().messages.batches.retrieve(state["batch_id"])
    c = b.request_counts
    print(f"{state['batch_id']}: {b.processing_status} | processing {c.processing}, succeeded {c.succeeded}, "
          f"errored {c.errored}, canceled {c.canceled}, expired {c.expired}")


def cmd_watch(args, df):
    """Print a progress line every minute until the batch ends (status checks are free). Ctrl+C to stop."""
    import time
    from datetime import datetime
    from src.complaints import get_client
    state = load_state(args.name)
    client = get_client()
    start = time.time()
    import anthropic
    while True:
        try:
            b = client.messages.batches.retrieve(state["batch_id"])
        except anthropic.APIConnectionError:      # a Wi-Fi blip shouldn't stop the watch
            print(f"{datetime.now():%H:%M:%S} connection lost, retrying in 60 s", flush=True)
            time.sleep(60)
            continue
        c = b.request_counts
        total = c.processing + c.succeeded + c.errored + c.canceled + c.expired
        done = total - c.processing
        bar = "#" * int(30 * done / total) + "-" * (30 - int(30 * done / total))
        print(f"{datetime.now():%H:%M:%S} [{bar}] {done:,}/{total:,} done ({done / total:.0%}) | "
              f"{b.processing_status} | errored {c.errored} | {(time.time() - start) / 60:.0f} min watching", flush=True)
        if b.processing_status == "ended":
            print(f"\nBatch finished. Next: python scripts/run_batch.py fetch --name {args.name}")
            return
        time.sleep(60)


def cmd_fetch(args, df):
    from src.complaints import get_client
    state = load_state(args.name)
    client = get_client()
    b = client.messages.batches.retrieve(state["batch_id"])
    if b.processing_status != "ended":
        sys.exit(f"Batch still {b.processing_status}; try again later.")

    done = bl.done_ids()
    saved, problems = 0, []
    with open(bl.LABELS_PATH, "a", encoding="utf-8") as f:
        for res in client.messages.batches.results(state["batch_id"]):   # any order: keyed by custom_id
            cid = res.custom_id
            if cid in done:
                continue
            if res.result.type != "succeeded":
                problems.append((cid, res.result.type))
                continue
            msg = res.result.message
            if msg.stop_reason != "end_turn":
                problems.append((cid, msg.stop_reason))
                continue
            try:
                text = next(blk.text for blk in msg.content if blk.type == "text")
                labels = json.loads(text)["labels"]
            except (StopIteration, json.JSONDecodeError, KeyError):
                problems.append((cid, "unreadable output"))
                continue
            rec = bl.result_record(cid, "succeeded", labels, (msg.usage.input_tokens, msg.usage.output_tokens),
                                   msg.stop_reason)
            f.write(json.dumps(rec) + "\n")
            saved += 1
    print(f"Saved {saved:,} new results to {bl.LABELS_PATH}. {len(problems)} not saved.")
    for cid, why in problems[:20]:
        print("  ", cid, why)
    if problems:
        print("Those are still 'missing'; submit again with a new --name to send only them.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["prepare", "dry-run", "submit", "status", "watch", "fetch"])
    p.add_argument("--scope", choices=["test", "full"], default="test")
    p.add_argument("--name", help="batch name (default: the scope)")
    p.add_argument("--yes", action="store_true", help="skip the typed confirmation")
    p.add_argument("--force", action="store_true", help="prepare: rebuild the cleaned text")
    args = p.parse_args()
    if args.command in ("status", "watch", "fetch") and not args.name:
        p.error("status, watch and fetch need --name")
    df = load_reviews()
    {"prepare": cmd_prepare, "dry-run": cmd_dry_run, "submit": cmd_submit,
     "status": cmd_status, "watch": cmd_watch, "fetch": cmd_fetch}[args.command](args, df)


if __name__ == "__main__":
    main()
