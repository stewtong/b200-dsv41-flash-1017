#!/usr/bin/env python3
"""Derive one sustained cell from retained request and telemetry records."""
import argparse
import math
import json
from pathlib import Path


def quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return {"n": 0, "p50": None, "min": None, "max": None, "p95": None, "p99": None}
    def nearest(fraction, minimum):
        return ordered[math.ceil(fraction * len(ordered)) - 1] if len(ordered) >= minimum else None
    return {"n": len(ordered), "p50": nearest(0.50, 1), "min": ordered[0], "max": ordered[-1],
            "p95": nearest(0.95, 20), "p99": nearest(0.99, 100)}


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def logical_telemetry(config, telemetry):
    if config["arguments"]["ranks"] != 8:
        for sample in telemetry:
            sample["native_covered_ranks"] = [sample["rank"]]
        return telemetry, {"status": "verified", "mode": "distinct_replica_ports", "identity": "port plus full native labels"}
    grouped = {}
    for sample in telemetry:
        grouped.setdefault(sample["sequence"], []).append(sample)
    normalized = []
    mode_counts = {}
    for sequence, rows in sorted(grouped.items()):
        engine_sets = []
        for sample in rows:
            engines = set()
            for item in sample.get("counters", []) + sample.get("gauges", []):
                value = item.get("labels", {}).get("engine")
                if value is not None and str(value).isdigit():
                    engines.add(int(value))
            engine_sets.append(engines)
        self_matched = all(engines == {sample["rank"]} for sample, engines in zip(rows, engine_sets))
        unique_engines = sorted(set().union(*engine_sets)) if engine_sets else []
        distinct_single = all(len(engines) == 1 for engines in engine_sets) and sum(len(engines) for engines in engine_sets) == len(set().union(*engine_sets))
        full_aggregates = [sample for sample, engines in zip(rows, engine_sets) if engines == set(range(8))]
        if self_matched and len(rows) == 8:
            for sample in rows:
                sample["native_covered_ranks"] = [sample["rank"]]
                sample["rank_identity_provenance"] = "native engine label matches target rank"
                normalized.append(sample)
            mode = "native_rank_labels_match_targets"
        elif distinct_single and unique_engines == list(range(8)):
            for sample, engines in zip(rows, engine_sets):
                sample["source_request_rank"] = sample["rank"]
                sample["rank"] = next(iter(engines))
                sample["native_covered_ranks"] = list(engines)
                sample["rank_identity_provenance"] = "unique native engine label deduplicated across pulls"
                normalized.append(sample)
            mode = "unique_native_engine_labels_deduplicated"
        elif len(full_aggregates) == len(rows) and len(rows) == 8:
            sample = dict(full_aggregates[0])
            sample["rank"] = "aggregated"
            sample["native_covered_ranks"] = list(range(8))
            sample["rank_identity_provenance"] = "one native aggregate export containing all engine labels; seven duplicate pulls removed"
            normalized.append(sample)
            mode = "full_aggregate_deduplicated"
        else:
            for sample, engines in zip(rows, engine_sets):
                sample["rank_evidence_inconclusive"] = True
                sample["native_covered_ranks"] = sorted(engines)
                normalized.append(sample)
            mode = "inconclusive_native_rank_identity"
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    complete = mode_counts and set(mode_counts) <= {"native_rank_labels_match_targets", "unique_native_engine_labels_deduplicated", "full_aggregate_deduplicated"}
    return normalized, {"status": "verified" if complete else "inconclusive", "sequence_modes": mode_counts,
                        "identity": "native engine labels plus full metric label sets; never header count alone"}


def ranked_values(sample, field, name):
    return sum(item["value"] for item in sample.get(field, []) if item["name"] == name)


def derive(cell_dir, additional_request_files=()):
    cell_dir = Path(cell_dir)
    config = json.loads((cell_dir / "config.json").read_text())
    requests = read_jsonl(cell_dir / "requests.jsonl")
    for value in additional_request_files:
        requests.extend(read_jsonl(Path(value)))
    telemetry, rank_coverage = logical_telemetry(config, read_jsonl(cell_dir / "telemetry.jsonl"))
    requests.sort(key=lambda row: (row["request_id"], row.get("terminal_wall_ns") or 0))
    deduplicated = {}
    for row in requests:
        if row["request_id"] in deduplicated:
            raise ValueError("duplicate request identity " + row["request_id"])
        deduplicated[row["request_id"]] = row
    requests = list(deduplicated.values())
    start = config["measurement_start_wall_ns"]
    end = config["measurement_end_wall_ns"]
    drain = config["drain_deadline_wall_ns"]
    cohort = [row for row in requests if start <= start_wall(row, config) < end]
    warm = [row for row in requests if start_wall(row, config) < start]
    measured_cohort_conversations = sorted({row["conversation"]["conversation_id"] for row in cohort})
    scheduled_ids = set(measured_cohort_conversations)
    successful_cohort = [row for row in cohort if row.get("success") and row["conversation"]["conversation_id"] in scheduled_ids]
    cohort_failures = [row for row in cohort if not row.get("success")]
    terminal_in_window = [row for row in requests if row.get("success") and row.get("terminal_wall_ns") is not None and start <= row["terminal_wall_ns"] < end]
    completion_tokens_in_window = sum((row.get("usage") or {}).get("completion_tokens", 0) for row in terminal_in_window)
    cohort_output = sum((row.get("usage") or {}).get("completion_tokens", 0) or 0 for row in successful_cohort)
    warm_successful = [row for row in warm if row.get("success")]
    warm_output_tokens = sum((row.get("usage") or {}).get("completion_tokens", 0) or 0 for row in warm_successful)
    warm_terminals = [row["terminal_wall_ns"] for row in warm_successful]
    drain_successful = [row for row in successful_cohort if row["terminal_wall_ns"] >= end]
    drain_output_tokens = sum((row.get("usage") or {}).get("completion_tokens", 0) or 0 for row in drain_successful)
    cohort_start_times = [start_wall(row, config) for row in cohort]
    cohort_terminals = [row.get("terminal_wall_ns") for row in cohort if row.get("terminal_wall_ns") is not None]
    cohort_elapsed = ((max(cohort_terminals) - min(cohort_start_times)) / 1e9 if cohort_terminals and cohort_start_times else None)
    measured_seconds = (end - start) / 1e9
    successful_conversations = set()
    for conversation_id in scheduled_ids:
        rows = [row for row in cohort if row["conversation"]["conversation_id"] == conversation_id]
        if rows and all(row.get("success") for row in rows):
            successful_conversations.add(conversation_id)
    trace = []
    all_scheduled = [row for row in requests]
    for sample in telemetry:
        observed = sample["observed_wall_ns"]
        client_unfinished = len({row["conversation"]["conversation_id"] for row in all_scheduled
                                 if start_wall(row, config) <= observed and (row.get("terminal_wall_ns") is None or row["terminal_wall_ns"] > observed)})
        client_inflight = sum(1 for row in all_scheduled if row.get("dispatch_wall_ns") is not None and row["dispatch_wall_ns"] <= observed and (row.get("terminal_wall_ns") is None or row["terminal_wall_ns"] > observed))
        trace.append({"observed_wall_ns": observed, "offset_seconds": sample["offset_seconds"], "rank": sample["rank"],
                      "client_unfinished_conversations": client_unfinished,
                      "client_inflight_turns": client_inflight,
                      "running": ranked_values(sample, "gauges", "vllm:num_requests_running"),
                      "waiting": ranked_values(sample, "gauges", "vllm:num_requests_waiting"),
                      "kv_cache_usage_perc": ranked_values(sample, "gauges", "vllm:kv_cache_usage_perc"),
                      "error": sample.get("error")})
    counter_identity = {}
    for sample in telemetry:
        if sample.get("rank_evidence_inconclusive"):
            continue
        for item in sample.get("counters", []):
            native_rank = item.get("labels", {}).get("engine", sample["rank"]) if config["arguments"]["ranks"] == 8 else sample["rank"]
            key = (native_rank, item["name"], json.dumps(item.get("labels", {}), sort_keys=True, separators=(",", ":")))
            counter_identity.setdefault(key, []).append((sample["observed_wall_ns"], item["value"]))
    deltas = {}
    resets = []
    missing = []
    for (rank, name, labels), values in sorted(counter_identity.items()):
        before = [value for value in values if value[0] <= start]
        after = [value for value in values if value[0] >= end]
        if not before or not after:
            missing.append({"rank": rank, "name": name, "labels": json.loads(labels)})
            continue
        initial = max(before)[1]
        final = min(after)[1]
        if final < initial:
            resets.append({"rank": rank, "name": name, "labels": json.loads(labels), "before": initial, "after": final})
        else:
            deltas[f"{rank}|{name}|{labels}"] = {"rank": rank, "name": name, "labels": json.loads(labels), "delta": final - initial,
                                       "before_ns": max(before)[0], "after_ns": min(after)[0]}
    def delta_sum(name):
        rows = [row for row in deltas.values() if row["name"] == name]
        return sum(row["delta"] for row in rows) if rows else None
    drafts = delta_sum("vllm:spec_decode_num_drafts_total")
    accepted = delta_sum("vllm:spec_decode_num_accepted_tokens_total")
    proposed = delta_sum("vllm:spec_decode_num_draft_tokens_total")
    prefix_hits = delta_sum("vllm:prefix_cache_hits_total")
    prefix_queries = delta_sum("vllm:prefix_cache_queries_total")
    post_end = [sample for sample in telemetry if sample["observed_wall_ns"] >= (max(cohort_terminals) if cohort_terminals else end)]
    latest_by_rank = {}
    for sample in post_end or telemetry:
        if config["arguments"]["ranks"] == 8:
            for native_rank in sample["native_covered_ranks"]:
                subset = dict(sample)
                subset["rank"] = native_rank
                for field in ("gauges", "counters"):
                    subset[field] = [item for item in sample.get(field, []) if item.get("labels", {}).get("engine") is None or int(item["labels"]["engine"]) == native_rank]
                latest_by_rank[native_rank] = subset
            if not sample.get("rank_evidence_inconclusive"):
                continue
        latest_by_rank[sample["rank"]] = sample
    last_running = sum(ranked_values(latest, "gauges", "vllm:num_requests_running") for latest in latest_by_rank.values())
    last_waiting = sum(ranked_values(latest, "gauges", "vllm:num_requests_waiting") for latest in latest_by_rank.values())
    expected_ranks = set(range(config["arguments"]["ranks"]))
    complete_terminal_rank_coverage = (rank_coverage["status"] == "verified" and set(latest_by_rank) == expected_ranks and all(
        not sample.get("error") and all(any(item["name"] == name for item in sample.get("gauges", []))
                                       for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"))
        and not sample.get("duplicate_series")
        for sample in latest_by_rank.values()))
    terminal_count = len({row["conversation"]["conversation_id"] for row in requests if row.get("terminal_wall_ns") is not None})
    scheduled_count = len({row["conversation"]["conversation_id"] for row in requests})
    all_scheduled_success = bool(requests) and all(row.get("success") for row in requests)
    completed_before_drain = all(row.get("terminal_wall_ns") is not None and row["terminal_wall_ns"] <= drain for row in requests)
    errors = [row for row in requests if not row.get("success")]
    expected_conversations = {f"c{index:06d}" for index in range(config["scheduled_conversations_total"])}
    actual_conversations = {row["conversation"]["conversation_id"] for row in requests}
    expected_turns = config["arguments"].get("turn_limit") or {"W1": 8, "W2": 4}[config["arguments"]["workload"]]
    complete_scheduled_population = actual_conversations == expected_conversations and len(requests) == config["scheduled_conversations_total"] * expected_turns
    candidate = all_scheduled_success and completed_before_drain and complete_scheduled_population and complete_terminal_rank_coverage and terminal_count == scheduled_count and last_running == 0 and last_waiting == 0
    reasons = []
    if not all_scheduled_success:
        reasons.append("failed invalid or unsent scheduled turns")
    if not completed_before_drain:
        reasons.append("work exceeded drain deadline")
    if last_running != 0 or last_waiting != 0:
        reasons.append("engine queue did not reach zero")
    if not complete_scheduled_population:
        reasons.append("scheduled conversation or turn population is incomplete")
    if not complete_terminal_rank_coverage:
        reasons.append("terminal rank coverage is incomplete")
    return {"schema": "sustained-derived-cell/v1", "arm": config["arguments"]["arm"],
            "workload": config["arguments"]["workload"], "rate": config["arguments"]["rate"],
            "repeat": config["arguments"]["repeat"], "trace_version": config["trace_version"],
            "measurement": {"scheduled_conversations": config["scheduled_conversations_total"],
                            "scheduled_requests": len(requests), "measured_cohort_requests": len(cohort),
                            "measured_cohort_conversations": len(scheduled_ids),
                            "successful_measured_cohort_requests": len(successful_cohort),
                            "successful_measured_cohort_conversations": len(successful_conversations),
                            "failed_or_unsent_measured_cohort_requests": len(cohort_failures),
                            "warmup_requests": len(warm),
                            "scheduled_per_second": len(scheduled_ids) / measured_seconds,
                            "started_per_second": len({row["conversation"]["conversation_id"] for row in cohort if row.get("dispatch_wall_ns")}) / measured_seconds,
                            "successful_conversations_per_second": len(successful_conversations) / measured_seconds,
                            "completed_requests_per_second": len(terminal_in_window) / measured_seconds},
            "throughput": {"successful_window_output_tokens": completion_tokens_in_window,
                           "successful_window_output_tokens_per_second": completion_tokens_in_window / measured_seconds,
                           "successful_window_output_tokens_per_second_per_gpu": completion_tokens_in_window / measured_seconds / 8,
                           "measured_cohort_output_tokens": cohort_output,
                           "warmup_output_tokens": warm_output_tokens,
                           "warmup_last_terminal_offset_seconds": (max(warm_terminals) - start) / 1e9 if warm_terminals else None,
                           "post_measurement_cohort_output_tokens": drain_output_tokens,
                           "post_measurement_cohort_completed_requests": len(drain_successful),
                           "cohort_first_scheduled_to_last_terminal_seconds": cohort_elapsed,
                           "measured_cohort_output_tokens_per_second": cohort_output / cohort_elapsed if cohort_elapsed and cohort_elapsed > 0 else None},
            "latency_seconds": {"request_completion": quantiles([row["request_completion_seconds"] for row in successful_cohort]),
                                "first_output": quantiles([row["first_output_seconds"] for row in successful_cohort]),
                                "conversation_dispatch_lag": quantiles([row["dispatch_lag_seconds"] for row in cohort if row.get("dispatch_lag_seconds") is not None]),
                                "measured_drain": max(0.0, (max(cohort_terminals) - end) / 1e9) if cohort_terminals else None},
            "stream_decode_tokens_per_second": quantiles([row["stream_decode_tokens_per_second"] for row in successful_cohort if row.get("stream_decode_tokens_per_second") is not None]),
            "completion_tiers_seconds": {str(tier): {"p95": quantiles([row["request_completion_seconds"] for row in successful_cohort])["p95"],
                                                       "verified_in_this_cell": len(successful_cohort) >= 20 and quantiles([row["request_completion_seconds"] for row in successful_cohort])["p95"] <= tier}
                                                 for tier in (30, 60)},
            "engine": {"counter_deltas": deltas, "counter_resets": resets, "missing_counter_bounds": missing,
                       "native_rank_coverage": rank_coverage,
                       "counter_sample_interval_seconds": sorted({round((row["after_ns"] - row["before_ns"]) / 1e9, 9) for row in deltas.values()}),
                       "counter_sample_brackets_ns": {key: {"before_ns": row["before_ns"], "after_ns": row["after_ns"], "interval_seconds": (row["after_ns"] - row["before_ns"]) / 1e9} for key, row in deltas.items()},
                       "draft_step_deltas": drafts, "accepted_token_deltas": accepted,
                       "proposed_draft_token_deltas": proposed,
                       "acceptance_length": 1 + accepted / drafts if drafts and accepted is not None else None,
                       "draft_token_acceptance_fraction": accepted / proposed if proposed and accepted is not None else None,
                       "prefix_cache_hit_fraction": prefix_hits / prefix_queries if prefix_queries else None,
                       "preemptions_delta": delta_sum("vllm:num_preemptions_total")},
            "queue_backlog_trace": trace,
            "sustainability": {"candidate_for_repeat_reconciliation": candidate, "reasons": reasons,
                               "all_scheduled_turns_successful_by_drain": all_scheduled_success and completed_before_drain,
                               "client_terminal_conversations": terminal_count, "client_scheduled_conversations": scheduled_count,
                               "latest_engine_running": last_running, "latest_engine_waiting": last_waiting,
                               "complete_scheduled_population": complete_scheduled_population,
                               "complete_terminal_rank_coverage": complete_terminal_rank_coverage,
                               "repeat_required": True, "status": "candidate" if candidate else "not_sustainable_in_window"},
            "failures": [{"request_id": row["request_id"], "turn": row["turn"], "status": row.get("status"),
                          "error": row.get("error") or row.get("validation_error")} for row in errors]}


def start_wall(row, config):
    return config["start_wall_ns"] + int(row["scheduled_offset_seconds"] * 1e9)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cell_dir")
    p.add_argument("--additional-request-file", action="append", default=[])
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = derive(args.cell_dir, args.additional_request_file)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
