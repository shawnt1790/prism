"""Run harness x config x tasks x repeats (mini-SWE-agent harness).

Usage (from the repo root, with Docker running):
    python -m eval.run --config configs/none.yaml --tasks tasks/checkpoint.txt
    python -m eval.run --config configs/none.yaml --tasks tasks/checkpoint.txt \
        --model anthropic/claude-haiku-4-5-20251001 --repeats 1

For each run it writes:
    logs/<run_id>.jsonl            Prism events (run_start ... result), the contract format
    logs/<run_id>.traj.json        mini-SWE-agent's own trajectory, for debugging
    logs/preds/<config>-r<N>.json  patches in SWE-bench format, for scoring `solved` later
    logs/<run_id>.error.txt        only if the run crashed

Runs whose log already exists are skipped, so a crashed batch can simply be restarted.
Use --overwrite to redo them.

`solved` is written as null: it comes from the separate SWE-bench evaluation step.
"""

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# exit statuses in the contract (docs/design.md, RunResult)
SUBMITTED, STEP_LIMIT, COST_LIMIT, ERROR = "submitted", "step_limit", "cost_limit", "error"


# ---------- config and tasks ----------

def load_configs(config_path: str | Path) -> tuple[dict, dict, str]:
    """Returns (base settings, policy config, config name). base.yaml is shared by every config."""
    config_path = Path(config_path)
    base = yaml.safe_load((config_path.parent / "base.yaml").read_text()) or {}
    policy_cfg = yaml.safe_load(config_path.read_text()) or {}
    return base, policy_cfg, config_path.stem


def read_tasks(path: str | Path) -> list[str]:
    """One task ID per line; blank lines and # comments are ignored."""
    ids = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line.split()[0])
    return ids


def prism_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT).returncode != 0
        return commit + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def litellm_model_name(model: str) -> str:
    """configs use bare names (claude-haiku-5-5); litellm needs a provider prefix."""
    return model if "/" in model else f"anthropic/{model}"


def mini_swebench_config(base: dict, model: str) -> dict:
    """mini-SWE-agent's built-in SWE-bench config with our step limit and model.

    Its own cost_limit is disabled: PrismAgent enforces base.yaml's cost_limit_usd instead."""
    from minisweagent.config import builtin_config_dir, get_config_from_spec

    config = get_config_from_spec(str(builtin_config_dir / "benchmarks" / "swebench.yaml"))
    config["agent"]["step_limit"] = int(base.get("step_limit", 100))
    config["agent"]["cost_limit"] = 0
    config["model"]["model_name"] = litellm_model_name(model)
    return config


# ---------- policy ----------

def get_compress(policy_cfg: dict, run_id: str, logger):
    """Returns the function PrismAgent calls on each ToolResultEvent."""
    from adapters.mini_swe_agent import passthrough

    policy = policy_cfg.get("policy", "none")
    if policy == "none":
        return passthrough
    # Integration point with Person A's library (to agree on day 2):
    # prism.policy.make_compress(policy_cfg, run_id=..., logger=...) -> Callable[[ToolResultEvent], str]
    # It should build the policy from the registry, and write a CompressionRecord per call.
    try:
        from prism.policy import make_compress
    except ImportError as e:
        raise SystemExit(f"Config uses policy {policy!r}, but prism.policy.make_compress "
                         f"doesn't exist yet ({e}). Use configs/none.yaml for baselines.")
    return make_compress(policy_cfg, run_id=run_id, logger=logger)


# ---------- one run ----------

def run_one(*, instance: dict, config_name: str, policy_cfg: dict, mini_config: dict,
            repeat: int, cost_limit: float, logs_dir: Path, env=None, model=None) -> dict:
    """Run one task once. `env` and `model` can be injected (used by the tests);
    otherwise the task's Docker container and the configured model are created."""
    from minisweagent import __version__ as harness_version

    from adapters.mini_swe_agent import HARNESS_NAME, PrismAgent
    from prism.events import RunResult, RunStart
    from prism.log import JsonlLogger

    task_id = instance["instance_id"]
    run_id = f"{HARNESS_NAME}-{config_name}-{task_id}-r{repeat}"
    model_label = litellm_model_name(mini_config["model"]["model_name"]).split("/")[-1]
    log_path = logs_dir / f"{run_id}.jsonl"
    traj_path = logs_dir / f"{run_id}.traj.json"

    agent, exit_raw, submission = None, "error", ""
    start = time.time()
    with JsonlLogger(log_path) as logger:
        logger.log(RunStart(run_id=run_id, task_id=task_id, harness=HARNESS_NAME,
                            config=config_name, model=model_label, repeat=repeat,
                            harness_version=harness_version, prism_commit=prism_commit()))
        try:
            if env is None:
                from minisweagent.run.benchmarks.swebench import get_sb_environment
                env = get_sb_environment(mini_config, instance)
            if model is None:
                from minisweagent.models import get_model
                model = get_model(config=mini_config.get("model", {}))
            agent_cfg = {k: v for k, v in mini_config.get("agent", {}).items() if k != "agent_class"}
            agent_cfg["output_path"] = traj_path   # mini-SWE-agent saves its trajectory every step
            agent = PrismAgent(model, env, run_id=run_id, prism_logger=logger,
                               compress=get_compress(policy_cfg, run_id, logger),
                               model_name=model_label, prism_cost_limit=cost_limit, **agent_cfg)
            info = agent.run(instance["problem_statement"])
            exit_raw = info.get("exit_status") or "error"
            submission = info.get("submission") or ""
        except Exception as e:
            exit_raw = type(e).__name__
            (logs_dir / f"{run_id}.error.txt").write_text(traceback.format_exc())
            print(f"  ERROR in {run_id}: {e}")
        finally:
            if env is not None and hasattr(env, "cleanup"):
                env.cleanup()   # stop the Docker container

        if exit_raw == "Submitted":
            exit_status = SUBMITTED
        elif exit_raw == "LimitsExceeded" and agent is not None:
            exit_status = COST_LIMIT if agent.prism_hit_cost_limit else STEP_LIMIT
        else:
            exit_status = ERROR

        total_cost = None
        if agent is not None and agent.prism_cost_complete:
            total_cost = round(agent.prism_total_cost, 6)
        result = RunResult(
            run_id=run_id, task_id=task_id, harness=HARNESS_NAME, config=config_name,
            model=model_label, repeat=repeat,
            solved=None,   # filled in by the SWE-bench evaluation step
            exit_status=exit_status,
            total_steps=agent.n_calls if agent else 0,
            n_model_calls=agent.prism_n_model_calls if agent else 0,
            total_cost_usd=total_cost,
            wall_clock_s=round(time.time() - start, 1),
        )
        logger.log(result)

    _update_preds(logs_dir / "preds" / f"{config_name}-r{repeat}.json", task_id,
                  f"prism-{config_name}", submission)
    return {"run_id": run_id, "exit_status": exit_status, "steps": result.total_steps,
            "cost_usd": total_cost, "wall_clock_s": result.wall_clock_s, "raw_exit": exit_raw}


def _update_preds(path: Path, instance_id: str, model_name: str, patch: str) -> None:
    """SWE-bench predictions file: {instance_id: {model_name_or_path, instance_id, model_patch}}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[instance_id] = {"model_name_or_path": model_name, "instance_id": instance_id,
                         "model_patch": patch}
    path.write_text(json.dumps(data, indent=2))


# ---------- CLI ----------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="e.g. configs/none.yaml")
    p.add_argument("--tasks", default="tasks/checkpoint.txt", help="file with one task ID per line")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--model", help="override base.yaml's model, e.g. anthropic/claude-haiku-4-5-20251001")
    p.add_argument("--logs-dir", default="logs")
    p.add_argument("--overwrite", action="store_true", help="redo runs whose log already exists")
    args = p.parse_args(argv)

    base, policy_cfg, config_name = load_configs(args.config)
    for key, value in (base.get("env") or {}).items():   # e.g. MSWEA_COST_TRACKING
        os.environ.setdefault(key, str(value))

    model = args.model or base["model"]
    mini_config = mini_swebench_config(base, model)
    task_ids = read_tasks(args.tasks)
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    dataset = base.get("dataset", "princeton-nlp/SWE-bench_Verified")
    print(f"Loading {dataset} ({base.get('split', 'test')})...")
    instances = {row["instance_id"]: row for row in load_dataset(dataset, split=base.get("split", "test"))}
    missing = [t for t in task_ids if t not in instances]
    if missing:
        raise SystemExit(f"Not in {dataset}: {missing}")

    print(f"config={config_name} model={litellm_model_name(model)} tasks={len(task_ids)} repeats={args.repeats}")
    summaries = []
    for task_id in task_ids:
        for repeat in range(1, args.repeats + 1):
            run_id = f"mini-{config_name}-{task_id}-r{repeat}"
            log_path = logs_dir / f"{run_id}.jsonl"
            if log_path.exists():
                if not args.overwrite:
                    print(f"skip {run_id} (log exists; use --overwrite to redo)")
                    continue
                log_path.unlink()
            print(f"run  {run_id}")
            s = run_one(instance=instances[task_id], config_name=config_name, policy_cfg=policy_cfg,
                        mini_config=mini_config, repeat=repeat,
                        cost_limit=float(base.get("cost_limit_usd", 0) or 0), logs_dir=logs_dir)
            print(f"     -> {s['exit_status']} ({s['raw_exit']}), {s['steps']} steps, "
                  f"cost {s['cost_usd']}, {s['wall_clock_s']}s")
            summaries.append(s)
    print(f"Done: {len(summaries)} runs. Next: python -m eval.metrics")


if __name__ == "__main__":
    main()