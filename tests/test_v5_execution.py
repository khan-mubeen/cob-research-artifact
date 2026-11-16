"""unit tests for the backend/profile abstraction, the result store and the generated job scripts.
runnable directly (python ...) or under pytest.

nothing here loads a model, touches a network or needs an accelerator. the two questions these
tests exist to answer are: does a new launch refuse to guess its hardware, and does anything that
already worked on this machine now need a flag it did not need before
"""

import ast
import json
import re
import tempfile
from pathlib import Path

import config
from src.experiments.v5 import execution, model_registry, results_store, slurm

ROOT = Path(config.PROJECT_ROOT)
PROTOCOL_DIR = ROOT / "reports" / "epoch-2" / "production" / "protocol"
GATES_DIR = ROOT / "reports" / "epoch-2" / "production" / "gates"


def raises(fn, exc, fragment=""):
    try:
        fn()
    except exc as e:
        assert fragment in str(e), f"expected {fragment!r} in {e}"
        return
    raise AssertionError(f"expected {exc.__name__}")


def generate(backend, profile, out_dir, extra=()):
    argv = ["--backend", backend, "--profile", profile, "--profile-root", "/scratch/test_root",
            "--account", "test_account", "--out-dir", str(out_dir), *extra]
    slurm.main(argv)
    return sorted(Path(out_dir).glob("train_*.sh"))


# ── the two axes ────────────────────────────────────────────────────────────

def test_backend_and_profile_are_independent_axes():
    for backend in execution.BACKEND_CHOICES:
        for profile in execution.PROFILE_CHOICES:
            resolved = execution.resolve(backend, profile, root="/scratch/anywhere")
            assert resolved.backend.name == backend and resolved.profile.name == profile
    # the cell nothing runs today but that must stay structurally possible
    rocm_local = execution.resolve("rocm", "local")
    assert rocm_local.backend.name == "rocm" and rocm_local.profile.name == "local"
    assert execution.resolve("cuda", "roihu", root="/scratch/x").backend.default_optim == "adamw_8bit"


def test_backend_settings_and_profile_settings_do_not_leak_into_each_other():
    rocm_local = execution.resolve("rocm", "local")
    rocm_lumi = execution.resolve("rocm", "lumi", root="/scratch/x")
    cuda_lumi = execution.resolve("cuda", "lumi", root="/scratch/x")
    # the backend decides the optimizer wherever it runs
    assert rocm_local.backend.default_optim == rocm_lumi.backend.default_optim == "adamw_torch"
    # the profile decides the queues whatever the backend
    assert cuda_lumi.profile.partitions == rocm_lumi.profile.partitions
    assert cuda_lumi.backend.default_optim != rocm_lumi.backend.default_optim


def test_a_real_launch_refuses_to_guess_backend_or_profile():
    raises(lambda: execution.require(None, None, "launcher"), execution.ExecutionError, "--backend and --profile")
    raises(lambda: execution.require("cuda", None, "launcher"), execution.ExecutionError, "--profile must be given")
    raises(lambda: execution.require(None, "lumi", "launcher"), execution.ExecutionError, "--backend must be given")
    raises(lambda: execution.resolve("cuda", "lumi"), execution.ExecutionError, "has no root")
    raises(lambda: execution.resolve("tpu", "local"), execution.ExecutionError, "unknown backend")


def test_the_hugging_face_cache_comes_from_the_profile_and_local_is_left_alone():
    lumi = execution.resolve("rocm", "lumi", root="/scratch/x")
    job_env = lumi.job_environment("${COB_PROFILE_ROOT}")
    assert job_env["HF_HOME"] == "${COB_PROFILE_ROOT}/hf-cache"
    assert job_env["HF_HUB_CACHE"] == "${COB_PROFILE_ROOT}/hf-cache/hub"
    # a local run must not have its cache redirected, or every existing command re-downloads
    assert execution.resolve("cuda", "local").job_environment("${COB_PROFILE_ROOT}") .get("HF_HOME") is None


def test_no_site_path_is_hardcoded_in_the_training_or_measurement_code():
    """site paths belong to the profile configuration, never to the code that trains or measures"""
    for relative in ("src/experiments/finetune_lora.py", "src/experiments/v5/evaluate.py",
                     "src/experiments/v5/inference_profile.py", "src/experiments/v5/results_store.py",
                     "src/experiments/v5/training_data.py"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "/scratch/" not in text, relative
        assert "/projappl" not in text, relative
        assert "HF_HOME" not in text, f"{relative} sets or reads HF_HOME itself"


# ── nothing that already worked may need a new flag ─────────────────────────

def test_every_pre_existing_command_keeps_running_without_new_flags():
    """the new axes are additive: no existing entry point gained a required argument"""
    for relative in ("src/experiments/finetune_lora.py", "src/experiments/eval_finetune.py",
                     "src/experiments/eval_reverse_auction.py", "src/experiments/v5/validate.py",
                     "src/experiments/v5/sample_check.py"):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
                name = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else ""
                required = any(k.arg == "required" and getattr(k.value, "value", False) for k in node.keywords)
                assert not (name in ("--backend", "--profile") and required), f"{relative} now requires {name}"
        assert "execution.require(" not in (ROOT / relative).read_text(encoding="utf-8"), relative


def test_the_new_epoch_adapter_flag_is_optional_and_changes_nothing_by_default():
    tree = ast.parse((ROOT / "src" / "experiments" / "finetune_lora.py").read_text(encoding="utf-8"))
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" \
                and node.args and getattr(node.args[0], "value", "") == "--save-epoch-adapters":
            found = True
            assert any(k.arg == "action" and k.value.value == "store_true" for k in node.keywords)
            assert not any(k.arg in ("required", "default") and getattr(k.value, "value", None) not in (None, False)
                           for k in node.keywords)
    assert found, "--save-epoch-adapters is not declared"
    source = (ROOT / "src" / "experiments" / "finetune_lora.py").read_text(encoding="utf-8")
    assert "if args.save_epoch_adapters and not timing_only:" in source


# ── generated job scripts ───────────────────────────────────────────────────

def test_generated_scripts_call_the_one_shared_training_path():
    with tempfile.TemporaryDirectory() as tmp:
        scripts = generate("rocm", "lumi", tmp)
        # six models x (two bidder modes + supervisor)
        assert len(scripts) == len(model_registry.MODELS) * (len(slurm.TRAIN_MODES) + 1) == 18
        for model in model_registry.MODELS:
            assert (Path(tmp) / f"train_{model}_{slurm.SUPERVISOR_MODE}.sh").exists(), model
        assert (Path(tmp) / "submit_all.sh").exists()
        for path in scripts:
            text = path.read_text(encoding="utf-8")
            assert "src/experiments/finetune_lora.py" in text
            assert "--save-epoch-adapters" in text and "--num-train-epochs 3" in text
            assert "--resume" in text and "--save-steps 250" in text
            assert "set +e" in text and "status=$?" in text          # the wall-time path must be reachable
            assert "sbatch --dependency=afterany" in text            # continuation across wall times
            assert "C:\\" not in text and "\\m4\\" not in text       # no path from this machine
            assert "${COB_PROFILE_ROOT}" in text
            # the storage hierarchy, in the generated paths themselves: one run directory per
            # attempt, and the trainer's resume state under it rather than beside the adapters
            model, _, mode = path.stem[len("train_"):].partition("_")
            run = slurm.run_name(model, mode)
            run_dir = f"${{COB_PROFILE_ROOT}}/models/training/{mode}/{model}/{run}"
            assert f'--output-dir "{run_dir}"' in text
            assert f'--checkpoint-dir "{run_dir}/checkpoints"' in text
            assert f'"{run_dir}/TRAINING_DONE"' in text
            assert "/models/v5/" not in text                          # the collapsed layout is gone


def test_a_generated_script_carries_this_models_own_frozen_flags():
    with tempfile.TemporaryDirectory() as tmp:
        generate("rocm", "lumi", tmp)
        qwen = (Path(tmp) / "train_qwen_sealed.sh").read_text(encoding="utf-8")
        granite = (Path(tmp) / "train_granite_sealed.sh").read_text(encoding="utf-8")
        phi = (Path(tmp) / "train_phi_sealed.sh").read_text(encoding="utf-8")
        # a marker ending in a newline must survive as an ansi-c quoted argument
        assert "$'<|im_start|>user\\n'" in qwen
        assert "'<|start_of_role|>user<|end_of_role|>'" in granite
        assert "qkv_proj,o_proj,gate_up_proj,down_proj" in phi
        # lumi pins base_layout="profile" on its execution profile, so every model -- the two
        # derived local copies included -- reads from the materialized base tree, and a directory
        # carries no hub revision to pin
        assert '"${COB_PROFILE_ROOT}/models/base/phi"' in phi
        assert "--revision" not in phi
        # the registry layout is still what a site without a base tree renders
        with tempfile.TemporaryDirectory() as other:
            generate("cuda", "roihu", other, extra=["--partition", "gpu"])
            roihu_phi = (Path(other) / "train_phi_sealed.sh").read_text(encoding="utf-8")
            assert '"${COB_PROFILE_ROOT}/models/base-checkpoints/hf/phi4mini-v2-base"' in roihu_phi


def test_the_roihu_stub_runs_the_identical_experiment_logic():
    """the future-site proof: the experiment is the same everywhere, only siting differs

    two things are legitimately per-site and are excluded here rather than asserted equal: the
    optimizer default, which is a backend property, and where the weights are read from, which is
    a profile property now that lumi has a materialized base tree and roihu does not. everything
    that defines WHAT is trained has to be byte-identical across the two"""
    experiment = ("--dataset", "--max-seq-length", "--batch-size", "--grad-accum",
                  "--instruction-part", "--response-part", "--lora-targets", "--save-steps",
                  "--num-train-epochs")
    with tempfile.TemporaryDirectory() as lumi_dir, tempfile.TemporaryDirectory() as roihu_dir:
        generate("rocm", "lumi", lumi_dir)
        generate("cuda", "roihu", roihu_dir, extra=["--partition", "gpu"])
        lumi = (Path(lumi_dir) / "train_qwen_sealed.sh").read_text(encoding="utf-8")
        roihu = (Path(roihu_dir) / "train_qwen_sealed.sh").read_text(encoding="utf-8")
        line = lambda text: next(l for l in text.splitlines() if "finetune_lora.py" in l)  # noqa: E731
        def flag(text, name):
            parts = line(text).split()
            return parts[parts.index(name) + 1]
        for name in experiment:
            assert flag(lumi, name) == flag(roihu, name), name
        assert "--save-epoch-adapters" in lumi and "--save-epoch-adapters" in roihu
        assert "--resume" in lumi and "--resume" in roihu
        assert "--optim adamw_torch" in lumi and "--optim adamw_8bit" in roihu


def test_the_lumi_job_enters_the_container_and_runs_training_inside_it():
    """the environment that actually makes unsloth work on this site, asserted where it is emitted

    each of these cost a real debugging session to find, so each is pinned rather than trusted:
    the container module replaces singularity-AI-bindings (the two conflict), $SIF is checked
    because it stays silently empty on the wrong load order, HF_XET_CACHE is the one that keeps
    the xet cache out of a 22G home quota, and the compiler shim has to be built INSIDE the
    container because singularity resets PATH"""
    with tempfile.TemporaryDirectory() as tmp:
        text = (Path(generate("rocm", "lumi", tmp)[0])).read_text(encoding="utf-8")

        # never LOADED. a substring check would also trip on the comment that explains why the
        # job purges first, so this asserts the thing that actually matters: no load line
        assert not re.search(r"^module load singularity-AI-bindings", text, re.M)
        # and the job purges whatever the submitting shell had, because sbatch propagates it and
        # the container module conflicts with those same bindings
        assert re.search(r"^module purge$", text, re.M)
        purge_at = text.index("\nmodule purge\n")
        assert text.index("if ! type module") < purge_at < text.index("export EBU_USER_PREFIX")
        assert "module load LUMI/25.03 partition/container" in text
        assert "module load PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827" in text
        assert 'if [ -z "${SIF:-}" ]; then' in text           # empty $SIF must fail loudly

        assert 'export HF_XET_CACHE="${COB_PROFILE_ROOT}/hf-cache/xet"' in text
        assert 'export AITER_JIT_DIR="${COB_JOB_CACHE}/aiter"' in text
        assert 'export GPU_ARCHS="gfx90a"' in text

        # training runs inside the container, and the venv is sourced in there, never on the host.
        # the command is staged to a file first so it can be backgrounded: a synchronous heredoc
        # blocks this shell, and a blocked shell cannot service the wall-time signal
        assert 'cat > "$COB_INNER" <<' in text
        assert 'singularity exec "$SIF" bash -s < "$COB_INNER" &' in text
        inner = text.split("<<'COB_CONTAINER'", 1)[1].split("\nCOB_CONTAINER", 1)[0]
        assert "unsloth-env/bin/activate" in inner
        assert "finetune_lora.py" in inner
        # sourced ONLY inside the staged block. checking "before set +e" no longer means
        # anything now that the heredoc is staged earlier, so strip the heredoc and assert the
        # activate line does not survive as a host-level command
        host_only = text.replace(inner, "")
        assert "unsloth-env/bin/activate" not in host_only
        # the shim and its PATH are built inside, because singularity resets PATH on entry
        assert 'export PATH="${COB_JOB_CACHE}/bin:${PATH}"' in inner
        assert '"${COB_JOB_CACHE}/bin/c++"' in inner
        # and it fails before spending training time on an unusable venv
        assert 'python -c "import torch, unsloth"' in inner


def test_a_profile_with_no_container_runs_the_command_plainly():
    """roihu stages and backgrounds its command the same way, but enters no container

    the staging is not container-specific: every site needs it so the wall-time trap can run while
    training is in flight. what stays site-specific is the container entry and its environment"""
    with tempfile.TemporaryDirectory() as tmp:
        text = (Path(generate("cuda", "roihu", tmp, extra=["--partition", "gpu"])[0])
                ).read_text(encoding="utf-8")
        for absent in ("singularity exec", "unsloth-env", "AITER_JIT_DIR",
                       "HF_XET_CACHE", "module load LUMI/25.03"):
            assert absent not in text, absent
        assert 'cat > "$COB_INNER" <<' in text          # staged, like every site
        assert 'bash -s < "$COB_INNER" &' in text       # but run plainly, not in a container
        inner = text.split("<<'COB_CONTAINER'", 1)[1].split("\nCOB_CONTAINER", 1)[0]
        assert inner.strip().startswith("python src/experiments")


def test_every_generated_script_survives_the_wall_clock():
    """the continuation path that job 22194724 proved was broken

    Slurm SIGTERMs the batch shell at the wall clock, so a script that only checks the training
    command's exit status never resubmits. these four things together are the fix, and all of them
    have to be present or the run silently stops at the first wall time"""
    for backend, profile, extra in (("rocm", "lumi", ()), ("cuda", "roihu", ("--partition", "gpu"))):
        with tempfile.TemporaryDirectory() as tmp:
            for path in generate(backend, profile, tmp, extra=extra):
                text = path.read_text(encoding="utf-8")
                assert re.search(r"^#SBATCH --signal=B:USR1@\d+$", text, re.M), path.name
                assert "trap cob_on_walltime USR1" in text, path.name
                assert 'wait "$train_pid"' in text, path.name          # interruptible
                assert "COB_RESUBMITTED=0" in text, path.name          # one submit per job
                # the trap must be armed BEFORE training starts, or the signal is missed
                assert text.index("trap cob_on_walltime USR1") < text.index('cat > "$COB_INNER"')
                # and the continuation cap still applies on the signal path
                assert f'-ge "{slurm.MAX_CONTINUATIONS}"' in text


def test_a_job_asks_for_the_one_gpu_it_actually_uses():
    """finetune_lora.py is a single process; job 22194724 held 8 GCDs and used 1"""
    assert slurm.DEFAULT_GPUS == 1
    for backend, profile, extra in (("rocm", "lumi", ()), ("cuda", "roihu", ("--partition", "gpu"))):
        with tempfile.TemporaryDirectory() as tmp:
            text = (Path(generate(backend, profile, tmp, extra=extra)[0])).read_text(encoding="utf-8")
            assert "#SBATCH --gpus-per-node=1" in text, profile
        # an explicit request still wins, for a future multi-device launcher
        with tempfile.TemporaryDirectory() as tmp:
            text = (Path(generate(backend, profile, tmp, extra=(*extra, "--gpus", "4"))[0])
                    ).read_text(encoding="utf-8")
            assert "#SBATCH --gpus-per-node=4" in text, profile


def test_the_default_partition_is_one_a_three_epoch_run_fits():
    with tempfile.TemporaryDirectory() as tmp:
        generate("rocm", "lumi", tmp)
        text = (Path(tmp) / "train_qwen_sealed.sh").read_text(encoding="utf-8")
        assert "#SBATCH --partition=small-g" in text and "#SBATCH --time=72:00:00" in text
        # slurm does not expand variables in #SBATCH lines, so no script may rely on that
        assert not re.search(r"^#SBATCH .*\$\{", text, re.M)


# ── the result store ────────────────────────────────────────────────────────

def test_records_carry_the_full_provenance_set():
    record = results_store.provenance("qwen", "sealed", 1, "models/x/final", backend="cuda", profile="local")
    assert set(results_store.PROVENANCE_FIELDS) <= set(record)
    assert record["corpus_manifest_sha256"] and len(record["corpus_manifest_sha256"]) == 64
    assert record["hardware"]["host"] and "torch" in record["hardware"]
    assert re.fullmatch(r"[0-9a-f]{40}|unknown|git-not-available", record["git_commit"])


def test_appending_is_keyed_so_a_resumed_job_never_double_counts():
    with tempfile.TemporaryDirectory() as tmp:
        results = results_store.ResultSet("latency", "unit", root=tmp)
        for i in range(3):
            assert results.append({"record_key": f"k{i}", "value": i}) is True
        assert results.append({"record_key": "k1", "value": 99}) is False      # already done
        assert len(results.records()) == 3
        assert results.done_keys() == {"k0", "k1", "k2"}
        assert [r["value"] for r in results.records()] == [0, 1, 2]


def test_a_job_killed_mid_write_leaves_a_torn_line_that_is_never_parsed_or_duplicated():
    with tempfile.TemporaryDirectory() as tmp:
        results = results_store.ResultSet("latency", "unit", root=tmp)
        results.append({"record_key": "k0", "value": 0})
        results.append({"record_key": "k1", "value": 1})
        # simulate the kill: a partial json line with no newline, exactly what a killed writer leaves
        with open(results.path, "a", encoding="utf-8", newline="\n") as f:
            f.write('{"record_key": "k2", "val')
        assert results.torn_tail() is True
        assert len(results.records()) == 2                       # the torn tail is not parsed
        assert results.done_keys() == {"k0", "k1"}               # so k2 is redone, not skipped
        assert results.append({"record_key": "k2", "value": 2}) is True
        assert results.torn_tail() is False
        assert [r["record_key"] for r in results.records()] == ["k0", "k1", "k2"]
        assert len(results.records()) == 3                       # no duplicate, no corruption


def test_the_csv_is_rebuilt_from_the_jsonl_so_the_two_cannot_disagree():
    with tempfile.TemporaryDirectory() as tmp:
        results = results_store.ResultSet("latency", "unit", root=tmp)
        results.append({"record_key": "k0", "ms_per_decision": 12.5, "hardware": {"device": "x"}})
        results.append({"record_key": "k1", "ms_per_decision": 13.5, "hardware": {"device": "x"}})
        path = results.write_csv()
        rows = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 3 and "hardware.device" in rows[0]
        raises(lambda: results_store.ResultSet("bad name", "unit", root=tmp), ValueError, "simple name")


def test_latency_summary_separates_warmup_from_the_timed_rows():
    from src.experiments.v5 import inference_profile
    records = [{"warmup": True, "ms_per_decision": 900.0, "input_tokens": 500, "output_tokens": 40,
                "total_tokens": 540, "output_tokens_per_second": 44.0, "hit_token_cap": False},
               {"warmup": False, "ms_per_decision": 300.0, "input_tokens": 500, "output_tokens": 40,
                "total_tokens": 540, "output_tokens_per_second": 133.0, "hit_token_cap": False},
               {"warmup": False, "ms_per_decision": 320.0, "input_tokens": 520, "output_tokens": 44,
                "total_tokens": 564, "output_tokens_per_second": 137.0, "hit_token_cap": False}]
    summary = inference_profile.summarise(records, warmup=1)
    assert summary["rows_timed"] == 2 and summary["rows_warmup_excluded"] == 1
    assert summary["warmup_median_ms"] == 900.0                  # reported, not silently dropped
    assert summary["ms_per_decision"]["median"] == 310.0
    assert summary["output_tokens"]["median"] == 42.0            # real generated tokens, not the prompt


# ── the registry against the frozen protocol ────────────────────────────────

def test_the_model_registry_matches_the_frozen_protocol_files():
    for name in model_registry.MODELS:
        path = PROTOCOL_DIR / f"protocol_epoch2_{name}.json"
        if not path.exists():
            continue
        recorded = json.load(open(path, encoding="utf-8"))["training_frozen"]["base_checkpoints"][name]
        checkpoint, _, revision = recorded.partition(" @ ")
        spec = model_registry.model(name)
        assert spec["revision"] == revision.strip(), name
        assert Path(spec["checkpoint"]).name == Path(checkpoint.strip()).name, name
    frozen = model_registry.FROZEN
    assert (frozen["per_device_batch_size"], frozen["grad_accum"], frozen["save_steps"]) == (8, 2, 250)
    assert frozen["seed"] == 42 and frozen["lora_rank"] == 16


def test_the_registry_markers_match_the_recorded_template_gates():
    checked = 0
    for name in model_registry.MODELS:
        path = GATES_DIR / f"{name}_template_alignment.json"
        if not path.exists():
            continue
        gate = json.load(open(path, encoding="utf-8"))
        if "instruction_part" not in gate:
            continue
        spec = model_registry.model(name)
        assert spec["instruction_part"].strip() == gate["instruction_part"].strip(), name
        assert spec["response_part"].strip() == gate["response_part"].strip(), name
        checked += 1
    assert checked >= 1, "no template alignment gate carried explicit markers"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} v5 execution unit tests passed")


if __name__ == "__main__":
    main()
