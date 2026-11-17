#!/usr/bin/env python3
"""Sustained matched multi-turn streaming replay with explicit raw evidence."""
import argparse
import concurrent.futures
import copy
import datetime
import gzip
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import random
import shutil
import re
import threading
import time
import traceback
import urllib.request

import stream
import workload

TRACE_VERSION = "sustained-poisson-multiturn-realtext-v2"
PREPARED_LAYOUT_VERSION = "per-rate-profile-v2"
READ_ONLY_OLD_ROOT = Path("/data/hf-cache/sustained-speculation-layouts-20261116")
SCHEMA = "sustained-request-index/v1"
RATES = {"W1": [0.05, 0.15, 0.45, 1.35, 4.05],
         "W2": [0.02, 0.08, 0.32]}
WORKLOADS = {"W1": {"turns": 8, "base_tokens": 52000, "step_tokens": 3200},
             "W2": {"turns": 4, "base_tokens": 248000, "step_tokens": 3200}}
COUNTER_NAMES = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:request_success_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_preemptions_total",
)
GAUGE_NAMES = (
    "vllm:num_requests_running", "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


class Corpus:
    def __init__(self, code_path, prose_path):
        self.code = Path(code_path).read_text()
        self.prose = Path(prose_path).read_text()
        self.identity = {"code_sha256": sha256_bytes(Path(code_path).read_bytes()),
                         "prose_sha256": sha256_bytes(Path(prose_path).read_bytes())}
        if not self.code or not self.prose:
            raise ValueError("public corpus cannot be empty")

    def passage(self, selector, characters):
        source = self.code if sha256_bytes(selector.encode())[0] in "02468ace" else self.prose
        offset = int.from_bytes(hashlib.sha256(selector.encode()).digest()[:8], "big") % len(source)
        repeated = source + "\n" + source
        fragment = (repeated * (characters // len(repeated) + 2))[offset:offset + characters]
        return selector + "\n" + fragment


SYSTEM = (
    "You are a synthetic coding and document-analysis agent. Use the supplied public text only. "
    "Return controlled performance-test output. The transcript after each turn is fixed and does not "
    "depend on generated content."
)
TOOLS = [{"type": "function", "function": {
    "name": f"inspect_public_text_{index}",
    "description": "Inspect a bounded passage of public source code or public-domain prose.",
    "parameters": {"type": "object", "properties": {
        "selector": {"type": "string"}, "offset": {"type": "integer"},
    }, "required": ["selector", "offset"]}}} for index in range(12)]


def character_profile(corpus, workload_name):
    base = WORKLOADS[workload_name]
    system_characters = len(SYSTEM) + len(json.dumps(TOOLS))
    document_characters = max(1024, base["base_tokens"] * 4 - system_characters)
    return {"document_characters": document_characters,
            "assistant_characters": 1600,
            "addition_characters_by_turn": [(base["step_tokens"] - 400) * 4] * (base["turns"] - 1)}


def conversation_bodies(corpus, workload_name, rate, repeat, conversation_index, profile,
                        output_tokens=400, fixed_output=True, turn_limit=None):
    config = WORKLOADS[workload_name]
    identity = f"{TRACE_VERSION}|{workload_name}|{rate:.9f}|{repeat}|{conversation_index}"
    document = corpus.passage("document|" + identity, profile["document_characters"])
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Analyze this unique public document:\n" + document}]
    bodies = []
    turns = config["turns"] if turn_limit is None else turn_limit
    for turn in range(turns):
        seed = workload.seed_for(TRACE_VERSION + workload_name + str(rate), config["base_tokens"],
                                 1, repeat, conversation_index * config["turns"] + turn)
        body = workload.request_body("/model", copy.deepcopy(messages),
                                     config["base_tokens"] + config["step_tokens"] * turn,
                                     output_tokens, seed, thinking=False, fixed_output=fixed_output)
        body["tools"] = TOOLS
        body["tool_choice"] = "auto"
        bodies.append(body)
        if turn + 1 < turns:
            messages.append({"role": "assistant", "content": corpus.passage(
                f"fixed-assistant|{identity}|{turn}", profile["assistant_characters"])})
            messages.append({"role": "user", "content": "Fixed tool output for the next turn:\n" + corpus.passage(
                f"fixed-addition|{identity}|{turn}", profile["addition_characters_by_turn"][turn])})
    for previous, current in zip(bodies, bodies[1:]):
        if current["messages"][:len(previous["messages"])] != previous["messages"]:
            raise ValueError("an appended turn changed an earlier message prefix")
    return bodies


def tokenize(url, body):
    request_body = {key: copy.deepcopy(body[key]) for key in
                    ("model", "messages", "tools", "chat_template_kwargs") if key in body}
    request_body.update({"add_generation_prompt": True, "return_token_strs": False})
    payload = canonical(request_body)
    parsed = urllib.request.urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("tokenization must use loopback")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read())
    count = result.get("count")
    if not isinstance(count, int) or count <= 0:
        raise ValueError("tokenize response lacked a positive count")
    return {"count": count, "response_keys": sorted(result), "request_sha256": sha256_bytes(payload)}


def calibrate_profile(corpus, workload_name, tokenize_url, rate=None, repeat=0, initial_profile=None):
    profile = copy.deepcopy(initial_profile) if initial_profile is not None else character_profile(corpus, workload_name)
    calibration_rate = RATES[workload_name][0] if rate is None else rate
    config = WORKLOADS[workload_name]
    attempts = []
    history = {}
    keys = ["document_characters"] + [f"addition_{turn}" for turn in range(config["turns"] - 1)]
    seen_hashes = set()
    for key in keys:
        body_index = 0 if key == "document_characters" else int(key.split("_")[1]) + 1
        history[key] = []
        for iteration in range(8):
            bodies = conversation_bodies(corpus, workload_name, calibration_rate, repeat, 0, profile,
                                         turn_limit=body_index + 1)
            body = bodies[-1]
            request_hash = sha256_bytes(canonical(body))
            if request_hash in seen_hashes:
                raise ValueError("calibration failed to change the prompt")
            seen_hashes.add(request_hash)
            result = tokenize(tokenize_url, body)
            target = config["base_tokens"] + body_index * config["step_tokens"]
            current_characters = profile["document_characters"] if body_index == 0 else profile["addition_characters_by_turn"][body_index - 1]
            attempts.append({"key": key, "iteration": iteration, "target": target,
                             "observed": result["count"], "characters": current_characters})
            history[key].append((result["count"], current_characters))
            if abs(result["count"] - target) <= target * 0.0025:
                break
            if iteration == 7:
                raise ValueError(f"unable to calibrate {workload_name} turn {body_index}: " + json.dumps(attempts))
            if len(history[key]) >= 2 and history[key][-1][0] != history[key][-2][0] and history[key][-1][1] != history[key][-2][1]:
                previous_count, previous_characters = history[key][-2]
                current_characters = history[key][-1][1]
                adjustment = round((target - result["count"]) * (current_characters - previous_characters) / (result["count"] - previous_count))
            else:
                current_characters = history[key][-1][1]
                adjustment = round(current_characters * target / result["count"]) - current_characters
            adjustment = max(-history[key][-1][1] // 2, min(history[key][-1][1], adjustment))
            if body_index == 0:
                profile["document_characters"] = max(1024, profile["document_characters"] + adjustment)
            else:
                profile["addition_characters_by_turn"][body_index - 1] = max(256, current_characters + adjustment)
    return profile, attempts


def schedule_conversations(workload_name, rate, repeat, warmup_seconds, measured_seconds):
    seed = int.from_bytes(hashlib.sha256(canonical([TRACE_VERSION, workload_name, rate, repeat])).digest()[:8], "big")
    generator = random.Random(seed)
    rows = []
    arrival = 0.0
    while True:
        arrival += generator.expovariate(rate)
        if arrival >= warmup_seconds + measured_seconds:
            break
        rows.append({"conversation_index": len(rows), "conversation_id": f"c{len(rows):06d}",
                     "scheduled_offset_seconds": arrival,
                     "phase": "warmup" if arrival < warmup_seconds else "measured",
                     "arrival_seed_sha256": sha256_bytes(canonical([TRACE_VERSION, workload_name, rate, repeat]))})
    if not rows or not any(row["phase"] == "measured" for row in rows):
        raise ValueError("trace has an empty measured cohort")
    return rows


def prepare_inputs(args, schedules):
    corpus = Corpus(args.code_corpus, args.prose_corpus)
    rate_root = Path(args.shared_root) / "prepared" / TRACE_VERSION / PREPARED_LAYOUT_VERSION / args.workload / f"{args.rate:.9f}"
    prepared_root = rate_root / str(args.repeat)
    profiles_root = Path(args.shared_root) / "input-profiles" / TRACE_VERSION / PREPARED_LAYOUT_VERSION
    profile_path = profiles_root / f"{args.workload}.json"
    attempts = []
    reuse_records = []

    def reuse_immutable(relative, legacy=False):
        if legacy:
            source = READ_ONLY_OLD_ROOT / "prepared" / TRACE_VERSION / args.workload / f"{args.rate:.9f}" / str(args.repeat) / relative.name
        else:
            source = READ_ONLY_OLD_ROOT / relative.relative_to(Path(args.shared_root))
        if not source.exists() or source == relative:
            return
        destination = relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_bytes(source.read_bytes()) != sha256_bytes(destination.read_bytes()):
                raise ValueError("changed cached input identity")
            action = "identical_existing_cache"
        else:
            os.link(source, destination)
            action = "hardlink_immutable_old_bytes"
        if sha256_bytes(source.read_bytes()) != sha256_bytes(destination.read_bytes()):
            raise ValueError("immutable cache link did not preserve bytes")
        reuse_records.append({"source": str(source), "destination": str(destination),
                              "sha256": sha256_bytes(source.read_bytes()), "action": action})

    if args.repeat == 0:
        destination_dir = prepared_root
        destination_dir.mkdir(parents=True, exist_ok=True)
        for row in sorted(schedules, key=lambda item: item["conversation_index"]):
            legacy = args.workload == "W1" and args.rate == 0.05
            reuse_immutable(destination_dir / f"{row['conversation_id']}.json", legacy=legacy)
        if args.workload == "W1" and args.rate == 0.05:
            legacy_profile = READ_ONLY_OLD_ROOT / "input-profiles" / TRACE_VERSION / "W1.json"
            for destination in (profile_path, rate_root / "representative-profile.json"):
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(legacy_profile, destination) if not destination.exists() else None
                reuse_records.append({"source": str(legacy_profile), "destination": str(destination),
                                      "sha256": sha256_bytes(destination.read_bytes()), "action": "preserved_A005_input_mapping"})
        else:
            for destination in (profile_path, rate_root / "representative-profile.json"):
                reuse_immutable(destination)
    if profile_path.exists():
        base_profile = json.loads(profile_path.read_text())["profile"]
    else:
        if not args.tokenize_url:
            raise ValueError("first workload calibration requires --tokenize-url")
        base_profile, base_attempts = calibrate_profile(corpus, args.workload, args.tokenize_url)
        atomic_json(profile_path, {"schema": "sustained-input-profile/v1", "workload": args.workload,
                                   "profile": base_profile, "corpus": corpus.identity, "calibration": base_attempts,
                                   "created_utc": utc_now()})
    rate_profile_path = rate_root / "representative-profile.json"
    if rate_profile_path.exists():
        profile = json.loads(rate_profile_path.read_text())["profile"]
    elif args.rate == RATES[args.workload][0] and not profile_path.exists():
        raise ValueError("base profile was not retained")
    else:
        profile, attempts = calibrate_profile(corpus, args.workload, args.tokenize_url,
                                              rate=args.rate, initial_profile=base_profile)
        atomic_json(rate_profile_path, {"schema": "sustained-input-profile/v1", "workload": args.workload,
                                       "rate": args.rate, "profile": profile, "calibration": attempts,
                                       "created_utc": utc_now()})
        if args.rate == RATES[args.workload][0]:
            base_attempts = json.loads(profile_path.read_text())["calibration"]
            if profile == base_profile:
                attempts = base_attempts
                atomic_json(rate_profile_path, {"schema": "sustained-input-profile/v1", "workload": args.workload,
                                               "rate": args.rate, "profile": profile, "calibration": attempts,
                                               "created_utc": utc_now()})
    selected_rows = []
    for row in schedules:
        if args.client_count > 1 and row["conversation_index"] % args.client_count != args.client_index:
            continue
        selected_rows.append(row)

    def prepare_one(row):
        path = prepared_root / f"{row['conversation_id']}.json"
        tokenize_url = args.tokenize_balance_url if args.tokenize_balance_url and row["conversation_index"] % 2 else args.tokenize_url
        if path.exists():
            record = json.loads(path.read_text())
            if record["corpus"] != corpus.identity or record["trace_version"] != TRACE_VERSION:
                raise ValueError("prepared input identity mismatch")
        else:
            bodies = conversation_bodies(corpus, args.workload, args.rate, args.repeat,
                                         row["conversation_index"], profile,
                                         output_tokens=args.output_tokens, turn_limit=args.turn_limit)
            token_counts = [tokenize(tokenize_url, body) for body in bodies]
            record = {"schema": "sustained-prepared-conversation/v1", "trace_version": TRACE_VERSION,
                      "workload": args.workload, "rate": args.rate, "repeat": args.repeat,
                      "conversation": row, "corpus": corpus.identity, "profile_sha256": sha256_bytes(canonical(profile)),
                      "bodies": bodies, "tokenize": token_counts,
                      "request_sha256": [sha256_bytes(canonical(body)) for body in bodies],
                      "created_utc": utc_now()}
            atomic_json(path, record)
        record["scheduled_offset_seconds"] = row["scheduled_offset_seconds"]
        return record

    missing = [row for row in selected_rows if not (prepared_root / f"{row['conversation_id']}.json").exists()]
    if len(missing) > 1 and args.tokenize_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.tokenize_workers, len(missing)),
                                                  thread_name_prefix="bounded-tokenize") as executor:
            prepared = list(executor.map(prepare_one, selected_rows))
    else:
        prepared = [prepare_one(row) for row in selected_rows]
    conversations = list(prepared)
    return conversations, profile, attempts


def parse_metrics(text):
    counters, gauges = {}, {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)(?:\s+\d+)?", line)
        if not match:
            continue
        name, labels, raw_value = match.groups()
        if name not in COUNTER_NAMES + GAUGE_NAMES:
            continue
        label_values = {key: value for key, value in re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', labels or "")}
        key = (name, tuple(sorted(label_values.items())))
        try:
            value = float(raw_value)
        except ValueError:
            continue
        (counters if name in COUNTER_NAMES else gauges).setdefault(key, []).append(value)
    duplicate_series = []
    for population in (counters, gauges):
        for key, values in population.items():
            if len(values) != 1:
                duplicate_series.append({"name": key[0], "labels": dict(key[1]), "values": values})
    return {"counters": [{"name": key[0], "labels": dict(key[1]), "value": values[0]}
                         for key, values in sorted(counters.items())],
            "gauges": [{"name": key[0], "labels": dict(key[1]), "value": values[0]}
                       for key, values in sorted(gauges.items())],
            "duplicate_series": duplicate_series}


def metrics_worker(args, start_wall_ns, stop_event, raw_dir):
    path = Path(args.output_dir) / "telemetry.jsonl"
    sequence = 0
    with path.open("a") as output:
        while not stop_event.wait(max(0.0, start_wall_ns / 1e9 + sequence * 5 - time.time()) if sequence else 0):
            observed = time.time_ns()
            for rank in range(args.ranks):
                port = args.engine_ports[0] if len(args.engine_ports) == 1 else args.engine_ports[rank]
                url = f"http://127.0.0.1:{port}/metrics"
                headers = {"X-data-parallel-rank": str(rank)} if len(args.engine_ports) == 1 and args.ranks > 1 else {}
                sample = {"sequence": sequence, "observed_wall_ns": observed,
                          "offset_seconds": (observed - start_wall_ns) / 1e9, "rank": rank, "port": port}
                try:
                    request = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(request, timeout=10) as response:
                        text = response.read().decode()
                    if args.keep_raw_metrics:
                        destination = Path(raw_dir) / f"{sequence:06d}-rank-{rank}.txt.gz"
                        with gzip.open(destination, "wt") as raw:
                            raw.write(text)
                    sample.update(parse_metrics(text))
                    sample["raw_sha256"] = sha256_bytes(text.encode())
                except Exception as exc:
                    sample["error"] = type(exc).__name__ + ": " + str(exc)
                output.write(json.dumps(sample, allow_nan=False) + "\n")
                output.flush()
            sequence += 1


def write_raw(result, metadata, raw_root, index_path, lock):
    request_id = result["request_id"]
    relative = Path("requests") / (request_id + ".json.gz")
    destination = Path(raw_root) / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    with gzip.open(temporary, "wt") as output:
        json.dump(result, output, allow_nan=False, separators=(",", ":"))
    os.replace(temporary, destination)
    metadata["raw_path"] = str(relative)
    metadata["raw_sha256"] = sha256_bytes(destination.read_bytes())
    with lock:
        with Path(index_path).open("a") as index:
            index.write(json.dumps(metadata, allow_nan=False) + "\n")
            index.flush()


def conversation_worker(args, prepared, start_wall_ns, drain_wall_ns, cancellation, counter, lock):
    row = prepared["conversation"]
    session_id = f"sustspec-{args.workload}-{args.rate:.9f}-r{args.repeat}-{row['conversation_id']}"
    for turn, body in enumerate(prepared["bodies"]):
        if turn == 0:
            delay = start_wall_ns / 1e9 + prepared["scheduled_offset_seconds"] - time.time()
            if delay > 0:
                time.sleep(delay)
        scheduled_turn_wall_ns = time.time_ns() if turn == 0 else None
        if cancellation.cancelled or time.time_ns() >= drain_wall_ns:
            for missing_turn in range(turn, len(prepared["bodies"])):
                metadata = {"schema": SCHEMA, "request_id": f"{session_id}-t{missing_turn}",
                            "session_id": session_id, "conversation": row, "turn": missing_turn,
                            "arm": args.arm, "workload": args.workload, "rate": args.rate, "repeat": args.repeat,
                            "request_sha256": prepared["request_sha256"][missing_turn],
                            "scheduled_offset_seconds": prepared["scheduled_offset_seconds"],
                            "dispatch_wall_ns": None, "terminal_wall_ns": time.time_ns(),
                            "status": "unsent_drain_deadline", "success": False,
                            "client_index": args.client_index}
                with lock:
                    with (Path(args.output_dir) / "requests.jsonl").open("a") as index:
                        index.write(json.dumps(metadata) + "\n")
                        index.flush()
            return
        request_id = f"{session_id}-t{turn}"
        timeout = max(1.0, (drain_wall_ns - time.time_ns()) / 1e9 + 2)
        dispatch_wall_ns = time.time_ns()
        with lock:
            counter["dispatched"] += 1
        result = stream.capture(f"http://127.0.0.1:{args.router_port}/v1/chat/completions", body,
                                request_id, timeout, cancellation=cancellation, session_id=session_id)
        metadata = {"schema": SCHEMA, "request_id": request_id, "session_id": session_id,
                    "conversation": row, "turn": turn, "arm": args.arm, "workload": args.workload,
                    "rate": args.rate, "repeat": args.repeat, "request_sha256": prepared["request_sha256"][turn],
                    "actual_request_body_sha256": result["request_body_sha256"],
                    "prepared_tokenize": prepared["tokenize"][turn], "scheduled_offset_seconds": prepared["scheduled_offset_seconds"],
                    "scheduled_turn_wall_ns": scheduled_turn_wall_ns, "dispatch_wall_ns": dispatch_wall_ns,
                    "terminal_wall_ns": time.time_ns(), "client_index": args.client_index,
                    "dispatch_lag_seconds": max(0.0, (dispatch_wall_ns - start_wall_ns) / 1e9 - prepared["scheduled_offset_seconds"]) if turn == 0 else None,
                    "http_status": result["http_status"], "complete": result["complete"], "error": result["error"]}
        try:
            summary = stream.summarize_stream(result)
            usage = summary["usage"]
            completion_tokens = usage.get("completion_tokens")
            valid = (result["http_status"] == 200 and summary["finish_reason"] == "length" and
                     completion_tokens == args.output_tokens and usage.get("prompt_tokens") == prepared["tokenize"][turn]["count"])
            if not valid and args.allow_nonvalid_smoke:
                valid = result["http_status"] == 200 and result["complete"]
            metadata.update({"success": valid, "status": "complete" if valid else "invalid_contract",
                             "first_output_wall_ns": dispatch_wall_ns + summary["first_output_ns"],
                             "first_visible_wall_ns": dispatch_wall_ns + summary["first_visible_ns"] if summary["first_visible_ns"] is not None else None,
                             "last_output_wall_ns": dispatch_wall_ns + summary["last_output_ns"],
                             "finish_reason": summary["finish_reason"], "usage": usage,
                             "request_completion_seconds": (result["terminal_ns"] - result["start_ns"]) / 1e9,
                             "first_output_seconds": summary["first_output_ns"] / 1e9})
            delta = summary["last_output_ns"] - summary["first_output_ns"]
            metadata["stream_decode_tokens_per_second"] = ((completion_tokens - 1) / (delta / 1e9)
                                                              if delta > 0 and completion_tokens > 0 else None)
            expected_shape = WORKLOADS[args.workload]["base_tokens"] + WORKLOADS[args.workload]["step_tokens"] * turn
            metadata["earlier_message_prefixes_unchanged"] = True
            if (row["conversation_index"] == 0 and turn < 3 and args.repeat == 0 and
                    (not valid or abs(usage.get("prompt_tokens", 0) - expected_shape) > expected_shape * .0025)):
                metadata["success"] = False
                metadata["status"] = "invalid_representative_shape"
                with lock:
                    counter["qualification_failure"] = metadata["status"]
                cancellation.cancel()
        except Exception as exc:
            metadata.update({"success": False, "status": "invalid_stream", "validation_error": type(exc).__name__ + ": " + str(exc)})
        write_raw(result, metadata, Path(args.output_dir) / "raw", Path(args.output_dir) / "requests.jsonl", lock)
        with lock:
            counter["terminal"] += 1


def run_window(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    schedules = schedule_conversations(args.workload, args.rate, args.repeat, args.warmup_seconds, args.measured_seconds)
    conversations, profile, calibration = prepare_inputs(args, schedules)
    if args.client_safety_ceiling < 1:
        raise ValueError("client safety ceiling must be positive")
    if args.global_start_wall_ns:
        start_wall_ns = args.global_start_wall_ns
    else:
        start_wall_ns = time.time_ns() + 2_000_000_000
    drain_wall_ns = start_wall_ns + int((args.warmup_seconds + args.measured_seconds + args.drain_seconds) * 1e9)
    config = {"schema": "sustained-window-config/v1", "trace_version": TRACE_VERSION, "arguments": vars(args),
              "source_sha256": {name: sha256_bytes(Path(name).read_bytes()) for name in (__file__, stream.__file__, workload.__file__)},
              "corpus_sha256": Corpus(args.code_corpus, args.prose_corpus).identity,
              "profile": profile, "calibration_attempts": calibration,
              "scheduled_conversations_total": len(schedules), "scheduled_conversations_this_client": len(conversations),
              "scheduled_conversation_ids": [row["conversation_id"] for row in schedules],
              "start_wall_ns": start_wall_ns, "measurement_start_wall_ns": start_wall_ns + args.warmup_seconds * 1_000_000_000,
              "measurement_end_wall_ns": start_wall_ns + (args.warmup_seconds + args.measured_seconds) * 1_000_000_000,
              "drain_deadline_wall_ns": drain_wall_ns, "created_utc": utc_now()}
    atomic_json(Path(args.output_dir) / "config.json", config)
    cancellation = stream.Cancellation()
    stop_metrics = threading.Event()
    counter = {"dispatched": 0, "terminal": 0}
    lock = threading.Lock()
    metrics = None
    if not args.no_metrics:
        (Path(args.output_dir) / "raw" / "metrics").mkdir(parents=True, exist_ok=True)
        metrics = threading.Thread(target=metrics_worker, args=(args, start_wall_ns, stop_metrics,
                                                                 Path(args.output_dir) / "raw" / "metrics"), daemon=True)
        metrics.start()
    worker_limit = args.client_safety_ceiling // args.client_count if args.client_count > 1 else args.client_safety_ceiling
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(worker_limit, len(conversations)),
                                                      thread_name_prefix="bounded-conversation")
    futures = [executor.submit(conversation_worker, args, prepared, start_wall_ns, drain_wall_ns,
                               cancellation, counter, lock) for prepared in conversations]
    failure = None
    try:
        while time.time_ns() < config["measurement_end_wall_ns"] and not cancellation.cancelled:
            time.sleep(min(1.0, (config["measurement_end_wall_ns"] - time.time_ns()) / 1e9))
        while time.time_ns() < drain_wall_ns and not cancellation.cancelled and any(not future.done() for future in futures):
            time.sleep(0.5)
        if any(not future.done() for future in futures):
            cancellation.cancel()
        for future in futures:
            future.result()
        executor.shutdown(wait=True)
        if counter.get("qualification_failure"):
            failure = counter["qualification_failure"]
            Path(args.output_dir).joinpath("failure.txt").write_text(failure + "; representative warm-up qualification stopped the window\n")
        time.sleep(5.5)
    except BaseException as exc:
        failure = type(exc).__name__ + ": " + str(exc)
        cancellation.cancel()
        executor.shutdown(wait=True, cancel_futures=False)
        Path(args.output_dir).joinpath("failure.txt").write_text(failure + "\n" + traceback.format_exc())
    finally:
        stop_metrics.set()
        if metrics:
            metrics.join(timeout=20)
    atomic_json(Path(args.output_dir) / "window-state.json", {"terminal_utc": utc_now(),
                 "failure": failure, "counter": counter, "all_threads_terminal": all(future.done() for future in futures),
                 "actual_end_wall_ns": time.time_ns(), "drain_deadline_wall_ns": drain_wall_ns})
    import derive_cell
    atomic_json(Path(args.output_dir) / "summary.json", derive_cell.derive(args.output_dir))
    if failure:
        raise RuntimeError(failure)
    reuse_manifest = Path(args.shared_root) / "operations" / f"cache-reuse-{args.arm}-{args.workload}-{args.rate:.9f}-r{args.repeat}-client{args.client_index}.json"
    atomic_json(reuse_manifest, {"schema": "continuation-cache-reuse/v1", "records": reuse_records, "files": len(reuse_records),
                                "old_root_read_only": True, "repeat": args.repeat,
                                "aggregate_worker_ceiling": args.client_safety_ceiling * args.client_count,
                                "per_client_worker_limit": worker_limit})


def natural_stop(args):
    corpus = Corpus(args.code_corpus, args.prose_corpus)
    profile = character_profile(corpus, "W1")
    profile["document_characters"] = 1024
    body = conversation_bodies(corpus, "W1", RATES["W1"][0], 0, 0, profile,
                               output_tokens=128, fixed_output=False, turn_limit=1)[0]
    body["messages"] = [{"role": "user", "content": "Reply with exactly: public natural stop check passed"}]
    body.pop("tools", None)
    body.pop("tool_choice", None)
    request_id = f"sustspec-natural-stop-{args.arm}-{int(time.time())}"
    result = stream.capture(f"http://127.0.0.1:{args.router_port}/v1/chat/completions", body,
                            request_id, 300, session_id=request_id)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_raw(result, {"request_id": request_id, "arm": args.arm, "check": "natural_stop"},
              Path(args.output_dir) / "raw", Path(args.output_dir) / "requests.jsonl", threading.Lock())
    summary = stream.summarize_stream(result)
    content = "".join((choice.get("delta") or {}).get("content") or ""
                      for event in result["events"] if event["kind"] == "event"
                      for choice in event["data"].get("choices", []))
    receipt = {"complete": result["complete"], "finish_reason": summary["finish_reason"],
               "content": content, "usage": summary["usage"], "success": "public natural stop check passed" in content}
    atomic_json(Path(args.output_dir) / "natural-stop.json", receipt)
    if not receipt["success"]:
        raise RuntimeError("natural-stop known-answer check failed")


def forced_smoke(args):
    corpus = Corpus(args.code_corpus, args.prose_corpus)
    profile, attempts = calibrate_profile(corpus, args.workload, args.tokenize_url)
    bodies = conversation_bodies(corpus, args.workload, args.rate, args.repeat, 0, profile,
                                 output_tokens=args.output_tokens, turn_limit=1)
    counts = [tokenize(args.tokenize_url, body) for body in bodies]
    body = bodies[0]
    request_id = f"sustspec-forced-smoke-{args.arm}-{int(time.time())}"
    result = stream.capture(f"http://127.0.0.1:{args.router_port}/v1/chat/completions", body,
                            request_id, 900, session_id=request_id)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_raw(result, {"request_id": request_id, "arm": args.arm, "check": "forced_smoke"},
              Path(args.output_dir) / "raw", Path(args.output_dir) / "requests.jsonl", threading.Lock())
    summary = stream.summarize_stream(result)
    usage = summary["usage"]
    receipt = {"complete": result["complete"], "finish_reason": summary["finish_reason"], "usage": usage,
               "tokenize_count": counts[0]["count"],
               "success": summary["finish_reason"] == "length" and usage.get("completion_tokens") == args.output_tokens and usage.get("prompt_tokens") == counts[0]["count"]}
    atomic_json(Path(args.output_dir) / "forced-smoke.json", receipt)
    if not receipt["success"] or abs(counts[0]["count"] - args.base_tokens) > args.base_tokens * 0.03:
        raise RuntimeError("forced smoke qualification failed")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True)
    p.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    p.add_argument("--rate", required=True, type=float)
    p.add_argument("--repeat", required=True, type=int)
    p.add_argument("--router-port", required=True, type=int)
    p.add_argument("--tokenize-url", required=True)
    p.add_argument("--tokenize-balance-url")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shared-root", required=True)
    p.add_argument("--code-corpus", required=True)
    p.add_argument("--prose-corpus", required=True)
    p.add_argument("--engine-ports", required=True, type=lambda value: [int(x) for x in value.split(",")])
    p.add_argument("--ranks", required=True, type=int)
    p.add_argument("--warmup-seconds", type=int, default=60)
    p.add_argument("--measured-seconds", type=int, default=360)
    p.add_argument("--drain-seconds", type=int, default=120)
    p.add_argument("--output-tokens", type=int, default=400)
    p.add_argument("--base-tokens", type=int)
    p.add_argument("--step-tokens", type=int)
    p.add_argument("--turn-limit", type=int)
    p.add_argument("--client-index", type=int, default=0)
    p.add_argument("--client-count", type=int, default=1)
    p.add_argument("--global-start-wall-ns", type=int)
    p.add_argument("--no-metrics", action="store_true")
    p.add_argument("--keep-raw-metrics", action="store_true", default=True)
    p.add_argument("--client-safety-ceiling", type=int, default=256)
    p.add_argument("--tokenize-workers", type=int, default=2)
    p.add_argument("--natural-stop", action="store_true")
    p.add_argument("--forced-smoke", action="store_true")
    p.add_argument("--allow-nonvalid-smoke", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if not 1 <= args.tokenize_workers <= 2:
        raise ValueError("exact preparation is capped at two workers")
    if args.base_tokens is not None:
        if args.base_tokens < 128:
            raise ValueError("base token override is too small")
        WORKLOADS[args.workload]["base_tokens"] = args.base_tokens
    if args.step_tokens is not None:
        if args.step_tokens < 500:
            raise ValueError("step token override is too small")
        WORKLOADS[args.workload]["step_tokens"] = args.step_tokens
    if args.workload in RATES and not any(math.isclose(args.rate, value) for value in RATES[args.workload]):
        raise ValueError("rate is outside the frozen workload ladder")
    if args.client_count not in (1, 2) or not 0 <= args.client_index < args.client_count:
        raise ValueError("invalid split-client identity")
    if not ((args.ranks == 2 and len(args.engine_ports) == 2) or (args.ranks == 8 and len(args.engine_ports) == 1)):
        raise ValueError("engine ports must be two replicas or one DEP8 front")
    if args.natural_stop:
        natural_stop(args)
    elif args.forced_smoke:
        forced_smoke(args)
    else:
        run_window(args)


if __name__ == "__main__":
    main()
