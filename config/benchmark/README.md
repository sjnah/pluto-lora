# Paper benchmark configuration

`paper_main_v1.yaml` is the canonical no-argument configuration for the PLUTO
LoRA paper benchmark.  Edit the file and run `bash train.sh` from the PLUTO
root.  CLI flags are optional one-run overrides and do not mutate the YAML.

## Workflow controls

- `workflow.mode`: `train_and_evaluate`, `train_only`, or `evaluate_only`.
- `training.existing_checkpoint`: reuse, retrain, or reject a completed model.
- `training.resume_policy`: automatically resume staged training, force a fresh
  run identity, or require a resumable phase checkpoint.
- `selection`: arms, training seeds, and evaluation benchmarks.
- `checkpoint_overrides`: explicit completed checkpoints keyed by arm and seed.

The common feature-cache identity lives in the referenced training protocol
under `runtime.feature_cache_name`; it is not an arm-selection concern.

`evaluate_only` never invokes training.  With
`evaluation.require_completed_checkpoint: true`, it fails before simulation if
any selected trainable arm has no completed checkpoint.  Every invocation
writes a resolved manifest to `artifacts/benchmark_runs/` before doing work.

## Paper-table arm contract

The source-comparison arms `random_exact`, `rule_exact`, `loss_exact`,
`mpoc_exact`, and `llm_exact_off` share the same training protocol, hard-replay
pacing, and exact bucket quota sampler.  `uniform` is one continuous 12-epoch
run without curriculum phases.

`llm_capped_off` and `llm_capped_on` are a matched LLM method-bundle ablation.
They use the same capped weighted sampler, persistent exposure controls, and
near-duplicate group weighting.  Type routing is the intentional difference.
No non-LLM arm can enable type routing.

## Checkpoint override example

```yaml
checkpoint_overrides:
  uniform:
    "1": outputs/example/lora_checkpoints/merged_final_ema.ckpt
  llm_capped_on:
    "1": /absolute/path/to/merged_final_ema.ckpt
```

Relative paths are resolved from the PLUTO repository root.  Overrides are
always validated as existing files, including in dry-run mode.
