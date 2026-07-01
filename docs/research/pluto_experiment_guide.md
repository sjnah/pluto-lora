# PLUTO Fine-tuning Experiment Guide

이 문서는 현재 정리된 `llm-taxonomy` 구조 기준의 PLUTO fine-tuning 실행
요약이다. 예전 `9_create_scenario_filters_for_training.py` 기반
`rule_all_*`/`llm_train_*`/`percentile_*` 흐름은 legacy naming으로
보관되었고, 현재는 `uniform_*`, `llm_guided_*`, `rulebased_*`,
`lossrank_*` 필터를 사용한다.

## 현재 필터 생성 위치

필터 생성은 `llm-taxonomy` 루트에서 수행한다.

```bash
cd /home/sjnah/Workspace/development/scene-intrinsic-difficulty/llm-taxonomy
```

LLM 기반 curriculum 필터:

```bash
python scripts/experiments/pluto/create_llm_guided_filters.py \
  --train-input artifacts/combined_difficulty_scores_v9_train.jsonl \
  --val-input artifacts/combined_difficulty_scores_v9_val.jsonl \
  --output-dir artifacts/scenario_filters/pluto_llm_guided \
  --train-ratio 0.85 \
  --val-ratio 0.15
```

Rule-based benchmark 필터:

```bash
python scripts/experiments/pluto/create_rulebased_filters.py \
  --train-input artifacts/combined_difficulty_scores_v9_train.jsonl \
  --val-input artifacts/combined_difficulty_scores_v9_val.jsonl \
  --output-dir artifacts/scenario_filters/pluto_rulebased \
  --train-ratio 0.85 \
  --val-ratio 0.15
```

PLUTO 실행이 읽는 YAML은 `pluto/config/scenario_filter/`에 있어야 한다.
현재 repo에는 이미 `uniform_*`, `llm_guided_*`, `rulebased_*`,
`lossrank_*` YAML이 보존되어 있다.

## 주요 실행 스크립트

```bash
cd /home/sjnah/Workspace/development/scene-intrinsic-difficulty/pluto
```

LLM-guided 기반 uniform/curriculum 비교:

```bash
bash train.sh
```

Rule-based score 기반 uniform/curriculum 비교군:

```bash
bash train.sh rulebased
```

테스트 및 분석:

```bash
bash test.sh
bash analyze.sh
```

이전 개별 실행 스크립트(`run_uniform_finetune.sh`,
`run_curriculum_finetune.sh`)는 중복된 legacy wrapper로
`archive/legacy_scripts/training/`에 보관했다. 현재는 위의 root wrapper를
사용한다.

## 현재 필터 이름

Uniform baseline:

- `uniform_train_all`
- `uniform_val_all`

LLM-guided curriculum:

- `llm_guided_train_easy`
- `llm_guided_train_medium`
- `llm_guided_train_hard`
- `llm_guided_val_easy`
- `llm_guided_val_medium`
- `llm_guided_val_hard`

Rule-based benchmark:

- `rulebased_train_all`
- `rulebased_train_easy`
- `rulebased_train_medium`
- `rulebased_train_hard`
- `rulebased_val_all`
- `rulebased_val_easy`
- `rulebased_val_medium`
- `rulebased_val_hard`

## Legacy

다음 파일은 현재 실행 경로가 아니다.

- `llm-taxonomy/archive/legacy_scripts/filter_generation/8_create_scenario_filters.py`
- `llm-taxonomy/archive/legacy_scripts/filter_generation/9_create_scenario_filters_for_training.py`
- `pluto/archive/legacy_scripts/`
- `pluto/archive/legacy_configs/`

이전 가이드의 `rule_all_train.yaml`, `llm_train_low.yaml` 같은 이름은 최신
PLUTO 실행 스크립트와 맞지 않는다.
