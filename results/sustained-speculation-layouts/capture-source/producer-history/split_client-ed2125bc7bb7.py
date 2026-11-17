#!/usr/bin/env python3
"""Replay one eligible trace through two actual client processes."""
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import derive_cell
import replay


def split_arguments(arguments, root, client):
    result = list(arguments)
    option = result.index('--output-dir')
    result[option + 1] = str(Path(root) / f'client{client}')
    return result


def main():
    args = replay.parser().parse_args()
    if args.client_count != 1 or args.global_start_wall_ns:
        raise ValueError("split parent requires an unsplit new start")
    schedules = replay.schedule_conversations(args.workload, args.rate, args.repeat, args.warmup_seconds, args.measured_seconds)
    replay.prepare_inputs(args, schedules)
    root = Path(args.output_dir)
    start = time.time_ns() + 5_000_000_000
    commands = []
    for client in range(2):
        arguments = split_arguments(copy.deepcopy(sys.argv[1:]), args.output_dir, client)
        arguments += ["--client-index", str(client), "--client-count", "2", "--global-start-wall-ns", str(start)]
        if client:
            arguments += ["--no-metrics"]
        commands.append([sys.executable, str(Path(replay.__file__).resolve())] + arguments)
    logs = []
    processes = []
    cancelled = False
    try:
        for client, command in enumerate(commands):
            log = (root / f"client{client}.log").open("wb")
            logs.append(log)
            (root / f"client{client}").mkdir(parents=True, exist_ok=True)
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        def stop(signum, frame):
            nonlocal cancelled
            cancelled = True
            for process in processes:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError("split client failure: " + json.dumps(codes))
        if cancelled:
            raise RuntimeError("split client cancelled by coordinator")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for log in logs:
            log.close()
    config = json.loads((root / "client0" / "config.json").read_text())
    config["arguments"]["client_count"] = 2
    config["split_client_evidence"] = True
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copyfile(root / "client0" / "telemetry.jsonl", root / "telemetry.jsonl")
    shutil.copyfile(root / "client1" / "requests.jsonl", root / "requests-client1.jsonl")
    summary = derive_cell.derive(root, [root / "requests-client1.jsonl"])
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
