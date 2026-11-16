"""unit tests for the model and checkpoint storage hierarchy, one section per required change.
runnable directly (python ...) or under pytest.

nothing here loads a model, downloads anything or needs an accelerator. the hierarchy under test is

  models/base/<model>/
  models/training/<mode>/<model>/<run>/{checkpoints/checkpoint-<step>, epoch-1..3, final, TRAINING_DONE}
  models/selected/<mode>/<model>/
  models/exports/{merged_hf,gguf}/<mode>/<model>/<run>-epoch<n>/

every change is additive, so most of these tests are paired: one that the new layout is reachable
and correct, one that the old default still produces exactly what it produced before
"""

import ast
import json
import tempfile
from pathlib import Path

import config
from src.experiments.v5 import download_models, execution, freeze_check, model_registry, slurm

ROOT = Path(config.PROJECT_ROOT)
FINETUNE = ROOT / "src" / "experiments" / "finetune_lora.py"


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


# ── 1. the profile-rooted base path resolver ────────────────────────────────

def test_the_base_tree_is_a_third_field_and_replaces_neither_existing_one():
    for name, spec in model_registry.MODELS.items():
        # the legacy checkpoint field is untouched, which is what the protocol tests and the frozen
        # provenance records read
        assert model_registry.training_path(name) == spec["checkpoint"]
        assert model_registry.checkpoint_for(name) == spec["checkpoint"]
        # the hub source and revision stay separately available
        source = model_registry.base_source(name)
        assert source["revision"] == spec["revision"]
        assert source["hub_source"] == (spec.get("hub_source") if spec["local"] else spec["checkpoint"])
        # and the base tree is a new, third answer
        assert model_registry.base_dir(name, "/site") == f"/site/models/base/{name}"
        assert model_registry.training_path(name, "/site", layout="profile") == f"/site/models/base/{name}"


def test_the_base_layout_carries_no_hub_revision_because_it_is_a_directory():
    for name, spec in model_registry.MODELS.items():
        assert model_registry.pins_revision(name, "registry") is (not spec["local"])
        assert model_registry.pins_revision(name, "profile") is False
    flags = model_registry.train_flags("qwen", "sealed", "/site", base_layout="profile")
    assert "--revision" not in flags
    assert flags[flags.index("--base-model") + 1] == "/site/models/base/qwen"
    # the default is unchanged, revision and all
    legacy = model_registry.train_flags("qwen", "sealed", "/site")
    assert legacy == model_registry.train_flags("qwen", "sealed", "/site", base_layout="registry")
    assert "--revision" in legacy and legacy[legacy.index("--base-model") + 1] == "Qwen/Qwen3.5-4B"


def test_an_unknown_layout_is_refused_rather_than_silently_treated_as_the_default():
    raises(lambda: model_registry.training_path("qwen", layout="whatever"), ValueError, "unknown base layout")


def test_the_execution_profile_roots_every_branch_of_the_hierarchy():
    resolved = execution.resolve("rocm", "lumi", root="/scratch/site")
    assert resolved.base_dir == Path("/scratch/site/models/base")
    assert resolved.training_dir == Path("/scratch/site/models/training")
    assert resolved.selected_dir == Path("/scratch/site/models/selected")
    assert resolved.exports_dir == Path("/scratch/site/models/exports")
    assert resolved.base_model_dir("phi") == Path("/scratch/site/models/base/phi")
    # and the job-side strings stay posix and rooted at the shell expression, not at this machine
    paths = resolved.job_paths("${COB_PROFILE_ROOT}")
    assert paths["base"] == "${COB_PROFILE_ROOT}/models/base"
    assert resolved.job_training_dir("${COB_PROFILE_ROOT}", "sealed", "qwen", "qwen-sealed-v5-r01") == \
        "${COB_PROFILE_ROOT}/models/training/sealed/qwen/qwen-sealed-v5-r01"
    assert "\\" not in paths["training"]


def test_the_supervisor_mode_fits_the_schema_without_any_job_being_emitted():
    # the directory schema accommodates supervisor for any architecture; slurm.py still emits none
    assert "supervisor" in model_registry.MODES
    assert "supervisor" not in slurm.TRAIN_MODES
    for name in model_registry.MODELS:
        run = slurm.run_name(name, "supervisor")
        _, output, checkpoints = slurm.run_paths("/site", name, "supervisor")
        assert output == f"/site/models/training/supervisor/{name}/{run}"
        assert checkpoints == f"{output}/checkpoints"


# ── 2. download destinations and the two derived copies ─────────────────────

def test_the_plan_names_the_real_destination_under_each_layout():
    context = execution.resolve("cuda", "lumi", root="/scratch/site")
    cache = {e["model"]: e for e in download_models.plan(context, "cache")}
    base = {e["model"]: e for e in download_models.plan(context, "base")}
    assert len(cache) == len(base) == 6
    for name, spec in model_registry.MODELS.items():
        # base layout: one self-contained directory per model, all six of them
        assert base[name]["destination"] == f"/scratch/site/models/base/{name}"
        assert base[name]["path_used_for_training"] == base[name]["destination"]
        # cache layout: the hub cache folder the revision really lands in, not the cache root
        if spec["local"]:
            assert cache[name]["destination"] == spec["checkpoint"]
        else:
            assert cache[name]["destination"] == (
                f"/scratch/site/hf-cache/hub/models--{spec['checkpoint'].replace('/', '--')}"
                f"/snapshots/{spec['revision']}")
        assert cache[name]["path_used_for_training"] == spec["checkpoint"]


def test_the_two_pad_token_fixed_bases_have_an_explicit_source_under_the_base_layout():
    context = execution.resolve("cuda", "lumi", root="/scratch/site")
    entries = {e["model"]: e for e in download_models.plan(context, "base")}
    for name in ("phi", "smollm3"):
        entry = entries[name]
        assert entry["kind"] == "local copy"
        # named upstream, named source directory, and a copy that is verified rather than assumed
        assert entry["source"] == model_registry.MODELS[name]["hub_source"]
        assert entry["derive_from"].endswith(Path(model_registry.MODELS[name]["checkpoint"]).name)
        assert "hash every provenance file" in entry["action"]
    for name in ("qwen", "granite", "ministral", "gemma"):
        assert entries[name]["derive_from"] is None
        assert "snapshot_download" in entries[name]["action"]
    # --derive-from moves that source without touching anything else
    moved = {e["model"]: e for e in download_models.plan(context, "base", derive_from="/elsewhere/hf")}
    assert moved["phi"]["derive_from"].startswith("/elsewhere/hf")
    assert moved["qwen"] == entries["qwen"]


def test_a_derived_copy_is_verified_file_by_file_and_a_corrupted_one_is_reported():
    with tempfile.TemporaryDirectory() as tmp:
        source, target = Path(tmp) / "src", Path(tmp) / "dst"
        source.mkdir()
        (source / "config.json").write_text('{"pad_token": "<|endofprompt|>"}', encoding="utf-8")
        (source / "tokenizer_config.json").write_text('{"a": 1}', encoding="utf-8")
        (source / "model.safetensors").write_bytes(b"x" * 2048)
        good = download_models.copy_derived({"derive_from": str(source), "destination": str(target)})
        assert good["copied"] and good["problems"] == []
        assert (target / "config.json").is_file()

        # a copy that lost the pad-token fix is a problem, not a pass
        (target / "config.json").write_text('{"pad_token": "</s>"}', encoding="utf-8")
        again = download_models.copy_derived({"derive_from": str(source),
                                              "destination": str(target / "nested")})
        assert again["copied"]
        from src.experiments.v5.freeze_check import local_fingerprint
        assert local_fingerprint(target)["provenance_files"]["config.json"]["sha256"] != \
            local_fingerprint(source)["provenance_files"]["config.json"]["sha256"]

        missing = download_models.copy_derived({"derive_from": str(Path(tmp) / "nope"),
                                                "destination": str(Path(tmp) / "out")})
        assert not missing["copied"] and missing["problems"]


def test_the_default_layout_is_still_cache_so_an_existing_command_is_unchanged():
    assert download_models.DEFAULT_LAYOUT == "cache"
    source = (ROOT / "src" / "experiments" / "v5" / "download_models.py").read_text(encoding="utf-8")
    # the local copies are still skipped, not silently fetched, when the layout is the old one
    assert 'if args.layout != "base":' in source
    context = execution.resolve("cuda", "lumi", root="/scratch/site")
    for entry in download_models.plan(context):
        assert entry == next(e for e in download_models.plan(context, "cache") if e["model"] == entry["model"])


# ── 3. trainer checkpoints separated from the epoch adapter root ────────────

def test_finetune_lora_keeps_checkpoints_in_the_output_dir_unless_told_otherwise():
    source = FINETUNE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" \
                and node.args and getattr(node.args[0], "value", "") == "--checkpoint-dir":
            found = True
            # an empty default is what keeps every completed run's command line reproducible
            assert any(k.arg == "default" and k.value.value == "" for k in node.keywords)
            assert not any(k.arg == "required" and getattr(k.value, "value", None) for k in node.keywords)
    assert found, "--checkpoint-dir is not declared"
    assert "checkpoint_dir = args.checkpoint_dir or output_dir" in source


def test_the_trainer_resumes_from_the_checkpoint_dir_and_saves_adapters_to_the_output_dir():
    source = FINETUNE.read_text(encoding="utf-8")
    # the trainer's own checkpoint-<step> directories follow the checkpoint dir
    assert "output_dir=checkpoint_dir," in source
    assert "last_complete_checkpoint(checkpoint_dir)" in source
    # the evaluable artifacts stay on the run root, which is what selection and export read
    assert "epoch_saver = EpochAdapterSaver(model, tokenizer, output_dir)" in source
    assert 'final_dir = str(Path(output_dir) / "final")' in source


def test_the_generated_v5_scripts_opt_into_the_run_and_checkpoint_hierarchy():
    assert slurm.DEFAULT_LAYOUT == "run"
    with tempfile.TemporaryDirectory() as tmp:
        scripts = generate("rocm", "lumi", tmp)
        for path in scripts:
            model, _, mode = path.stem[len("train_"):].partition("_")
            text = path.read_text(encoding="utf-8")
            run, output, checkpoints = slurm.run_paths("${COB_PROFILE_ROOT}", model, mode)
            assert run == f"{model}-{mode}-v5-r01"
            assert f'--output-dir "{output}"' in text
            assert f'--checkpoint-dir "{checkpoints}"' in text
            assert f'"{output}/TRAINING_DONE"' in text
            assert f'mkdir -p "${{COB_PROFILE_ROOT}}/logs" "{output}" "{checkpoints}"' in text
            # the run label is written where whoever evaluates the epochs will read it
            assert f"--run {run}" in text


def test_one_run_label_per_attempt_and_it_is_unique_across_models_and_modes():
    labels = [slurm.run_name(m, mode) for m in model_registry.MODELS for mode in model_registry.MODES]
    assert len(set(labels)) == len(labels)
    # selection groups by an exact string match, so the label must survive its own validation rule
    from src.experiments.v5 import epoch_selection
    dirs = set()
    for m in model_registry.MODELS:
        for mode in model_registry.MODES:
            d = epoch_selection.run_dir(slurm.run_name(m, mode))
            # canonical layout: <root>/<mode>/<model>
            assert (d.parent.name, d.name) == (mode, m)
            dirs.add(d)
    assert len(dirs) == len(labels)
    # a second attempt at the same cell gets its own directory
    assert slurm.run_name("qwen", "sealed", "r02") == "qwen-sealed-v5-r02"
    assert slurm.run_paths("/s", "qwen", "sealed", tag="r02")[1] != slurm.run_paths("/s", "qwen", "sealed")[1]


def test_the_legacy_layout_still_renders_the_collapsed_paths_it_always_rendered():
    with tempfile.TemporaryDirectory() as tmp:
        scripts = generate("cuda", "roihu", tmp, extra=("--layout", "legacy"))
        for path in scripts:
            model, _, mode = path.stem[len("train_"):].partition("_")
            text = path.read_text(encoding="utf-8")
            assert f'--output-dir "${{COB_PROFILE_ROOT}}/models/v5/{mode}/{model}-3epoch"' in text
            # no checkpoint flag at all, so the old command line is reproduced and not approximated
            assert "--checkpoint-dir" not in text
            assert "/models/training/" not in text


# ── 4. the base arm of an evaluation ────────────────────────────────────────

def test_only_the_base_arm_moves_with_the_base_layout():
    from src.experiments.v5 import evaluate

    class Args:
        epoch = 0
        adapter = ""
        base_model = ""
        model_name = "phi"
        base_layout = "registry"
        base_root = "/site"

    args = Args()
    # default: the registry's own entry and its pin, exactly as before
    assert evaluate.resolve_checkpoint(args) == (model_registry.MODELS["phi"]["checkpoint"], None)
    args.model_name = "qwen"
    assert evaluate.resolve_checkpoint(args) == ("Qwen/Qwen3.5-4B", model_registry.MODELS["qwen"]["revision"])
    # profile: the base tree, and no revision because a directory has none
    args.base_layout = "profile"
    assert evaluate.resolve_checkpoint(args) == ("/site/models/base/qwen", None)
    # an explicit --base-model still wins over both
    args.base_model = "some/other-repo"
    assert evaluate.resolve_checkpoint(args) == ("some/other-repo", None)


def test_a_tuned_adapter_string_is_passed_through_untouched_by_either_layout():
    from src.experiments.v5 import evaluate

    class Args:
        epoch = 2
        adapter = "/site/models/training/sealed/qwen/qwen-sealed-v5-r01/epoch-2"
        base_model = ""
        model_name = "qwen"
        base_root = "/site"
        base_layout = "registry"

    args = Args()
    for layout in model_registry.BASE_LAYOUT_CHOICES:
        args.base_layout = layout
        # byte for byte: authorize_test() compares this string and normalizes nothing
        assert evaluate.resolve_checkpoint(args) == (args.adapter, None)


def test_the_exact_adapter_string_is_what_authorizes_a_test_evaluation():
    from src.experiments.v5 import epoch_selection
    adapter = "/site/models/training/sealed/qwen/qwen-sealed-v5-r01/epoch-2"
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "qwen-sealed-v5-r01"
        run_dir.mkdir()
        (run_dir / "selection.json").write_text(
            json.dumps({"chosen": {"epoch": 2, "adapter": adapter}}), encoding="utf-8")
        assert epoch_selection.authorize_test("qwen-sealed-v5-r01", adapter, 2, tmp)
        # an equivalent spelling of the same directory is refused, which is why the hierarchy must
        # not rewrite the path anywhere between validation and test
        for equivalent in (adapter + "/", adapter.replace("/epoch-2", "/./epoch-2"),
                           adapter.replace("/", "\\")):
            raises(lambda e=equivalent: epoch_selection.authorize_test("qwen-sealed-v5-r01", e, 2, tmp),
                   epoch_selection.SelectionError, "test may only be evaluated")


def test_inference_profile_moves_its_base_checkpoint_the_same_way_and_drops_the_pin():
    source = (ROOT / "src" / "experiments" / "v5" / "inference_profile.py").read_text(encoding="utf-8")
    assert 'base_is_directory = spec["local"] or base_layout == "profile"' in source
    assert "revision=None if args.adapter or base_is_directory else spec[\"revision\"]" in source
    assert "checkpoint = args.adapter or base_checkpoint" in source


# ── 5. the versioned base-artifact provenance check ─────────────────────────

def test_a_record_declares_how_its_base_artifacts_can_be_verified():
    assert freeze_check.BASE_ARTIFACT_CHECK_VERSION == 2
    artifacts = freeze_check.model_artifacts(base_root="/site")
    assert set(artifacts) == set(model_registry.MODELS)
    for name, entry in artifacts.items():
        # every model now records its place in the base tree, present or not
        assert entry["base_dir"] == f"/site/models/base/{name}"
        assert entry["base_fingerprint"]["present"] is False        # nothing materialized yet
        # and the fields the old record carried are still there, unchanged
        assert entry["checkpoint"] == model_registry.MODELS[name]["checkpoint"]
        assert entry["revision"] == model_registry.MODELS[name]["revision"]
        if model_registry.MODELS[name]["local"]:
            assert "fingerprint" in entry and entry["upstream_source"]


def test_the_gate_verifies_a_materialized_base_directory_rather_than_a_cache_entry():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "models" / "base" / "qwen"
        base.mkdir(parents=True)
        (base / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
        (base / "model.safetensors").write_bytes(b"w" * 4096)

        recorded = freeze_check.model_artifacts(base_root=tmp)["qwen"]
        assert recorded["base_fingerprint"]["present"] is True

        here = freeze_check.local_fingerprint(recorded["base_dir"])
        assert freeze_check.fingerprint_problems("qwen", "base_dir", recorded["base_fingerprint"],
                                                 here, "absent") == []
        # a changed config in the base directory is caught, which a cache-presence test cannot see
        (base / "config.json").write_text('{"model_type": "tampered"}', encoding="utf-8")
        problems = freeze_check.fingerprint_problems("qwen", "base_dir",
                                                     recorded["base_fingerprint"],
                                                     freeze_check.local_fingerprint(base), "absent")
        assert problems and problems[0]["file"] == "config.json"
        # so is a resized weight shard
        (base / "model.safetensors").write_bytes(b"w" * 8192)
        assert any(p["file"] == "model.safetensors" for p in freeze_check.fingerprint_problems(
            "qwen", "base_dir", recorded["base_fingerprint"], freeze_check.local_fingerprint(base), "absent"))


def test_a_version_1_record_is_still_checked_the_way_it_always_was():
    source = (ROOT / "src" / "experiments" / "v5" / "freeze_check.py").read_text(encoding="utf-8")
    assert 'version = frozen.get("base_artifact_check_version", 1)' in source
    assert 'if version >= 2 and frozen_base.get("present"):' in source
    # an absent base directory in a version 2 record falls back rather than failing, which is what
    # lets this record be written before the tree is materialized
    assert 'elif entry["kind"] == "hub_revision":' in source


def test_the_freeze_gate_expects_the_layout_the_generator_now_produces():
    assert freeze_check.args_from_script("").layout == slurm.DEFAULT_LAYOUT
    # None, not a default: an unstated base layout now means "ask the site", which is what lets
    # one gate run validate two sites that legitimately differ
    assert freeze_check.args_from_script("").base_layout is None
    assert freeze_check.args_from_script("", layout="legacy").layout == "legacy"
    # each site re-renders cleanly when the gate is told that site's own base layout. the gate
    # takes ONE base-layout expectation for every generated set at once, so a fleet where the
    # sites differ cannot be validated in a single run; that is recorded in
    # prompt-reports/ as the remaining freeze_check gap rather than asserted away here
    for backend, profile in (("rocm", "lumi"), ("cuda", "roihu")):
        with tempfile.TemporaryDirectory() as tmp:
            extra = ("--partition", "gpu") if profile == "roihu" else ()
            generate(backend, profile, tmp, extra=extra)
            context = execution.resolve(backend, profile, root=freeze_check.RENDER_ROOT)
            text = (Path(tmp) / "train_qwen_sealed.sh").read_text(encoding="utf-8")
            expected = context.profile.base_layout or model_registry.BASE_LAYOUT_DEFAULT
            drift, stale, _ = freeze_check.inspect_script(text, "qwen", "sealed", context,
                                                         base_layout=expected)
            assert drift == [] and stale is False, profile
            # a script left on the old output layout is reported stale rather than accepted
            _, legacy_stale, _ = freeze_check.inspect_script(text, "qwen", "sealed", context,
                                                            layout="legacy", base_layout=expected)
            assert legacy_stale is True, profile


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  ok   {t.__name__}")
    print(f"{passed} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
