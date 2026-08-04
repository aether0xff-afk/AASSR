from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch_training() -> None:
    path = ROOT / "src/aassr_v2/escape_training.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "    imagination_minimum_coverage: float = 0.75\n",
        "    imagination_minimum_coverage: float = 0.35\n",
        label="escape coverage default",
    )
    text = replace_once(
        text,
        "    save_episode_checkpoints: bool = True\n",
        '''    save_episode_checkpoints: bool = True\n    checkpoint_interval: int = 100\n    checkpoint_retention: int = 10\n    step_flush_interval: int = 64\n    max_step_log_bytes: int = 1_073_741_824\n''',
        label="storage config fields",
    )
    text = replace_once(
        text,
        '''        if self.efficiency_bonus_scale < 0.0:\n            raise ValueError("efficiency_bonus_scale must be non-negative")\n''',
        '''        if self.efficiency_bonus_scale < 0.0:\n            raise ValueError("efficiency_bonus_scale must be non-negative")\n        if self.checkpoint_interval <= 0:\n            raise ValueError("checkpoint_interval must be positive")\n        if self.checkpoint_retention < 0:\n            raise ValueError("checkpoint_retention must be non-negative")\n        if self.step_flush_interval <= 0:\n            raise ValueError("step_flush_interval must be positive")\n        if self.max_step_log_bytes < 0:\n            raise ValueError("max_step_log_bytes must be non-negative")\n''',
        label="storage config validation",
    )
    text = replace_once(
        text,
        '''            if config.save_episode_checkpoints:\n                recorder.write_checkpoint(agent, episode=episode)\n''',
        '''            if config.save_episode_checkpoints and (\n                episode == 1\n                or episode % config.checkpoint_interval == 0\n            ):\n                recorder.write_checkpoint(agent, episode=episode)\n''',
        label="periodic checkpoint scheduling",
    )
    path.write_text(text, encoding="utf-8")


def patch_reporting() -> None:
    path = ROOT / "src/aassr_v2/escape_reporting.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''class EscapeSessionRecorder:\n    """Durable, flush-on-write recorder for every escape session event."""\n''',
        '''class EscapeSessionRecorder:\n    """Durable recorder with bounded full-trace and checkpoint storage."""\n''',
        label="recorder docstring",
    )
    text = replace_once(
        text,
        '''        self.mode_counts: Counter[str] = Counter({initial_mode: 1})\n        self._closed = False\n\n        self._steps_file = (self.output_dir / "steps.jsonl").open("w", encoding="utf-8", buffering=1)\n''',
        '''        self.mode_counts: Counter[str] = Counter({initial_mode: 1})\n        self._closed = False\n        self._step_flush_interval = int(\n            getattr(config, "step_flush_interval", 64)\n        )\n        self._max_step_log_bytes = int(\n            getattr(config, "max_step_log_bytes", 1_073_741_824)\n        )\n        self._checkpoint_retention = int(\n            getattr(config, "checkpoint_retention", 10)\n        )\n        self._pending_step_records = 0\n        self._step_records_written = 0\n        self._step_records_dropped = 0\n        self._step_bytes_written = 0\n        self._trace_truncated = False\n        self._checkpoint_files_pruned = 0\n\n        self._steps_file = (self.output_dir / "steps.jsonl").open(\n            "w", encoding="utf-8", buffering=1024 * 1024\n        )\n''',
        label="recorder storage state",
    )
    text = replace_once(
        text,
        '''            "config": _json_safe(config),\n            "world": serialize_world_spec(spec),\n            "files": {\n''',
        '''            "config": _json_safe(config),\n            "world": serialize_world_spec(spec),\n            "storage_policy": {\n                "step_flush_interval": self._step_flush_interval,\n                "max_step_log_bytes": self._max_step_log_bytes,\n                "checkpoint_retention": self._checkpoint_retention,\n            },\n            "files": {\n''',
        label="manifest storage policy",
    )
    old_record_step = '''    def record_step(self, payload: Mapping[str, Any]) -> None:\n        enriched = {\n            "schema_version": SCHEMA_VERSION,\n            "session_id": self.session_id,\n            **payload,\n        }\n        action = payload.get("action")\n        if isinstance(action, Mapping):\n            signature = str(action.get("signature", "unknown"))\n            self.action_counts[signature] += 1\n        event = str(payload.get("event", ""))\n        if event:\n            self.event_counts[event] += 1\n        self._steps_file.write(json.dumps(_json_safe(enriched), ensure_ascii=False) + "\\n")\n        self._steps_file.flush()\n\n'''
    new_record_step = '''    def flush_steps(self) -> None:\n        if self._pending_step_records <= 0:\n            return\n        self._steps_file.flush()\n        self._pending_step_records = 0\n\n    def storage_status(self) -> dict[str, Any]:\n        return {\n            "step_records_written": self._step_records_written,\n            "step_records_dropped": self._step_records_dropped,\n            "step_bytes_written": self._step_bytes_written,\n            "max_step_log_bytes": self._max_step_log_bytes,\n            "trace_truncated": self._trace_truncated,\n            "checkpoint_retention": self._checkpoint_retention,\n            "checkpoint_files_pruned": self._checkpoint_files_pruned,\n            "retained_episode_checkpoints": len(\n                tuple(self.checkpoints_dir.glob("episode_*.json.gz"))\n            ),\n        }\n\n    def record_step(self, payload: Mapping[str, Any]) -> None:\n        enriched = {\n            "schema_version": SCHEMA_VERSION,\n            "session_id": self.session_id,\n            **payload,\n        }\n        action = payload.get("action")\n        if isinstance(action, Mapping):\n            signature = str(action.get("signature", "unknown"))\n            self.action_counts[signature] += 1\n        event = str(payload.get("event", ""))\n        if event:\n            self.event_counts[event] += 1\n\n        if self._trace_truncated:\n            self._step_records_dropped += 1\n            return\n        line = json.dumps(_json_safe(enriched), ensure_ascii=False) + "\\n"\n        encoded_size = len(line.encode("utf-8"))\n        if (\n            self._max_step_log_bytes > 0\n            and self._step_bytes_written + encoded_size\n            > self._max_step_log_bytes\n        ):\n            self._trace_truncated = True\n            self._step_records_dropped += 1\n            self.flush_steps()\n            self.log(\n                "full step trace reached max_step_log_bytes; "\n                "continuing with episode summaries and bounded checkpoints"\n            )\n            return\n\n        self._steps_file.write(line)\n        self._step_bytes_written += encoded_size\n        self._step_records_written += 1\n        self._pending_step_records += 1\n        if self._pending_step_records >= self._step_flush_interval:\n            self.flush_steps()\n\n'''
    text = replace_once(
        text,
        old_record_step,
        new_record_step,
        label="bounded step recorder",
    )
    text = replace_once(
        text,
        '''    def record_episode(self, record: EscapeEpisodeRecord) -> None:\n        self.records.append(record)\n''',
        '''    def record_episode(self, record: EscapeEpisodeRecord) -> None:\n        self.flush_steps()\n        self.records.append(record)\n''',
        label="episode boundary step flush",
    )
    old_checkpoint = '''    def write_checkpoint(self, agent: object, *, episode: int, final: bool = False) -> Path:\n        payload = serialize_agent_checkpoint(agent, episode=episode)\n        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")\n        path = self.checkpoints_dir / (\n            "final.json.gz" if final else f"episode_{episode:06d}.json.gz"\n        )\n        with gzip.open(path, "wb", compresslevel=6) as handle:\n            handle.write(encoded)\n        latest = self.checkpoints_dir / "latest.json.gz"\n        with gzip.open(latest, "wb", compresslevel=6) as handle:\n            handle.write(encoded)\n        return path\n\n'''
    new_checkpoint = '''    def _prune_episode_checkpoints(self) -> None:\n        checkpoints = sorted(self.checkpoints_dir.glob("episode_*.json.gz"))\n        excess = max(0, len(checkpoints) - self._checkpoint_retention)\n        for path in checkpoints[:excess]:\n            path.unlink(missing_ok=True)\n            self._checkpoint_files_pruned += 1\n\n    def write_checkpoint(self, agent: object, *, episode: int, final: bool = False) -> Path:\n        payload = serialize_agent_checkpoint(agent, episode=episode)\n        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")\n        path = self.checkpoints_dir / (\n            "final.json.gz" if final else f"episode_{episode:06d}.json.gz"\n        )\n        with gzip.open(path, "wb", compresslevel=6) as handle:\n            handle.write(encoded)\n        latest = self.checkpoints_dir / "latest.json.gz"\n        with gzip.open(latest, "wb", compresslevel=6) as handle:\n            handle.write(encoded)\n        if not final:\n            self._prune_episode_checkpoints()\n        return path\n\n'''
    text = replace_once(
        text,
        old_checkpoint,
        new_checkpoint,
        label="checkpoint retention",
    )
    text = replace_once(
        text,
        '''        final_payload = {\n            **_json_safe(summary),\n''',
        '''        self.flush_steps()\n        final_payload = {\n            **_json_safe(summary),\n''',
        label="final step flush",
    )
    text = replace_once(
        text,
        '''            "mode_selection_counts": dict(self.mode_counts),\n            "error": error,\n''',
        '''            "mode_selection_counts": dict(self.mode_counts),\n            "storage": self.storage_status(),\n            "error": error,\n''',
        label="summary storage status",
    )
    text = replace_once(
        text,
        '''            f"mean_duration_seconds: {statistics_payload['durations_seconds']['mean']:.6f}",\n            f"output_dir: {self.output_dir}",\n''',
        '''            f"mean_duration_seconds: {statistics_payload['durations_seconds']['mean']:.6f}",\n            f"trace_truncated: {int(self._trace_truncated)}",\n            f"step_records_written: {self._step_records_written}",\n            f"step_records_dropped: {self._step_records_dropped}",\n            f"output_dir: {self.output_dir}",\n''',
        label="summary text storage status",
    )
    text = replace_once(
        text,
        '''                "summary_file": "summary.json",\n            }\n''',
        '''                "summary_file": "summary.json",\n                "storage": self.storage_status(),\n            }\n''',
        label="manifest final storage status",
    )
    text = replace_once(
        text,
        '''        self._closed = True\n        for handle in (\n''',
        '''        self._closed = True\n        self.flush_steps()\n        for handle in (\n''',
        label="close step flush",
    )
    path.write_text(text, encoding="utf-8")


def patch_cli() -> None:
    path = ROOT / "scripts/run_escape_gridworld.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''    parser.add_argument(\n        "--no-episode-checkpoints",\n        action="store_true",\n        help="skip per-episode compressed recovery checkpoints; final checkpoint is still saved",\n    )\n''',
        '''    parser.add_argument(\n        "--no-episode-checkpoints",\n        action="store_true",\n        help="skip periodic compressed recovery checkpoints; final checkpoint is still saved",\n    )\n    parser.add_argument(\n        "--checkpoint-every",\n        type=int,\n        default=100,\n        help="save a recovery checkpoint every N completed episodes",\n    )\n    parser.add_argument(\n        "--checkpoint-retention",\n        type=int,\n        default=10,\n        help="retain at most this many historical episode checkpoints",\n    )\n    parser.add_argument(\n        "--step-flush-interval",\n        type=int,\n        default=64,\n        help="flush the full step JSONL after this many records; episodes always flush",\n    )\n    parser.add_argument(\n        "--max-step-log-gb",\n        type=float,\n        default=1.0,\n        help="maximum steps.jsonl size in GiB; 0 means unlimited",\n    )\n''',
        label="CLI storage options",
    )
    text = replace_once(
        text,
        '''    args = build_parser().parse_args()\n    if args.gui:\n''',
        '''    args = build_parser().parse_args()\n    if args.checkpoint_every <= 0:\n        raise SystemExit("--checkpoint-every must be positive")\n    if args.checkpoint_retention < 0:\n        raise SystemExit("--checkpoint-retention must be non-negative")\n    if args.step_flush_interval <= 0:\n        raise SystemExit("--step-flush-interval must be positive")\n    if args.max_step_log_gb < 0.0:\n        raise SystemExit("--max-step-log-gb must be non-negative")\n    if args.gui:\n''',
        label="CLI storage validation",
    )
    text = replace_once(
        text,
        '''        use_imagination=not args.no_imagination,\n        save_episode_checkpoints=not args.no_episode_checkpoints,\n    )\n''',
        '''        use_imagination=not args.no_imagination,\n        save_episode_checkpoints=not args.no_episode_checkpoints,\n        checkpoint_interval=args.checkpoint_every,\n        checkpoint_retention=args.checkpoint_retention,\n        step_flush_interval=args.step_flush_interval,\n        max_step_log_bytes=int(args.max_step_log_gb * 1024**3),\n    )\n''',
        label="CLI config plumbing",
    )
    path.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    path = ROOT / "tests/test_escape_gridworld.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''        minimum_holdout_count=2,\n        save_episode_checkpoints=True,\n    )\n''',
        '''        minimum_holdout_count=2,\n        save_episode_checkpoints=True,\n        checkpoint_interval=1,\n    )\n''',
        label="legacy checkpoint test interval",
    )
    insertion_anchor = '''\ndef test_descriptive_statistics_and_rolling_mean() -> None:\n'''
    new_tests = '''\ndef test_step_trace_is_bounded_without_stopping_training(tmp_path: Path) -> None:\n    output = tmp_path / "bounded-trace"\n    config = EscapeTrainingConfig(\n        episodes=2,\n        seed=19,\n        color_count=1,\n        distractor_boxes=0,\n        use_imagination=False,\n        live_step_delay=0.0,\n        minimum_holdout_count=2,\n        save_episode_checkpoints=False,\n        step_flush_interval=64,\n        max_step_log_bytes=512,\n    )\n\n    summary = train_escape_agent(\n        config,\n        mode=TrainingMode.FAST,\n        output_dir=output,\n    )\n    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))\n\n    assert summary.episodes == 2\n    assert saved["storage"]["trace_truncated"]\n    assert saved["storage"]["step_records_dropped"] > 0\n    assert (output / "steps.jsonl").stat().st_size <= 512\n    assert (output / "episodes.csv").exists()\n    assert (output / "checkpoints" / "final.json.gz").exists()\n\n\ndef test_periodic_checkpoints_keep_only_recent_history(tmp_path: Path) -> None:\n    output = tmp_path / "checkpoint-retention"\n    config = EscapeTrainingConfig(\n        episodes=4,\n        seed=19,\n        color_count=1,\n        distractor_boxes=0,\n        use_imagination=False,\n        live_step_delay=0.0,\n        minimum_holdout_count=2,\n        save_episode_checkpoints=True,\n        checkpoint_interval=1,\n        checkpoint_retention=2,\n    )\n\n    train_escape_agent(\n        config,\n        mode=TrainingMode.FAST,\n        output_dir=output,\n    )\n    history = sorted(\n        path.name for path in (output / "checkpoints").glob("episode_*.json.gz")\n    )\n    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))\n\n    assert history == ["episode_000003.json.gz", "episode_000004.json.gz"]\n    assert saved["storage"]["checkpoint_files_pruned"] == 2\n    assert saved["storage"]["retained_episode_checkpoints"] == 2\n    assert (output / "checkpoints" / "latest.json.gz").exists()\n    assert (output / "checkpoints" / "final.json.gz").exists()\n\n\ndef test_storage_policy_validation() -> None:\n    with pytest.raises(ValueError, match="checkpoint_interval"):\n        EscapeTrainingConfig(checkpoint_interval=0)\n    with pytest.raises(ValueError, match="checkpoint_retention"):\n        EscapeTrainingConfig(checkpoint_retention=-1)\n    with pytest.raises(ValueError, match="step_flush_interval"):\n        EscapeTrainingConfig(step_flush_interval=0)\n    with pytest.raises(ValueError, match="max_step_log_bytes"):\n        EscapeTrainingConfig(max_step_log_bytes=-1)\n\n'''
    text = replace_once(
        text,
        insertion_anchor,
        new_tests + insertion_anchor,
        label="bounded storage tests",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_training()
    patch_reporting()
    patch_cli()
    patch_tests()


if __name__ == "__main__":
    main()
