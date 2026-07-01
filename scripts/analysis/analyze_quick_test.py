#!/usr/bin/env python3
"""
Analyze quick test results by reading individual metric files directly
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

def find_batch_directories(base_exp_dir):
    """
    Find all batch subdirectories for an experiment (e.g., experiment_batch0, experiment_batch1, etc.)
    
    Returns:
        List of batch directory paths, sorted by batch number
    """
    import glob
    import re
    
    base_name = os.path.basename(base_exp_dir)
    parent_dir = os.path.dirname(base_exp_dir)
    
    # Pattern: base_name_batch0, base_name_batch1, etc.
    pattern = os.path.join(parent_dir, f"{base_name}_batch*")
    batch_dirs = glob.glob(pattern)
    
    # Sort by batch number
    def get_batch_num(path):
        match = re.search(r'_batch(\d+)$', path)
        return int(match.group(1)) if match else -1
    
    return sorted(batch_dirs, key=get_batch_num)


def calculate_interplan_score(aggregator_metric_dir):
    """
    Calculate interPlan benchmark score from interPlan's aggregated metric file.
    interPlan uses its own metric aggregator that saves results to:
    aggregator_metric/closed_loop_reactive_agents_weighted_average_metrics_*.parquet
    
    Args:
        aggregator_metric_dir: Directory containing interPlan's aggregated metric file
        
    Returns:
        Dictionary with interPlan score information (similar format to NR-CLS for compatibility)
    """
    import glob
    
    if not os.path.exists(aggregator_metric_dir):
        return {
            'has_nr_cls': False,
            'error': f'Aggregator metric directory not found: {aggregator_metric_dir}'
        }
    
    # Find interPlan's aggregated metric file
    metric_files = glob.glob(os.path.join(aggregator_metric_dir, 'closed_loop_reactive_agents_weighted_average_metrics_*.parquet'))
    
    if not metric_files:
        return {
            'has_nr_cls': False,
            'error': f'No interPlan metric file found in: {aggregator_metric_dir}'
        }
    
    # Use the most recent file if multiple exist
    metric_file = sorted(metric_files)[-1]
    
    try:
        df = pd.read_parquet(metric_file)
        
        # Get final score (scenario == 'final_score')
        final_score_row = df[df['scenario'] == 'final_score']
        
        if final_score_row.empty:
            return {
                'has_nr_cls': False,
                'error': 'No final_score row found in interPlan metric file'
            }
        
        final_score = final_score_row.iloc[0]['score']
        num_scenarios = int(final_score_row.iloc[0]['num_scenarios'])
        
        # Get per-scenario scores (exclude final_score and scenario_type rows)
        scenario_rows = df[(df['scenario'] != 'final_score') & (df['scenario_type'] != 'final_score')]
        scenario_scores = scenario_rows['score'].tolist()
        
        # Calculate statistics
        score_mean = float(final_score)
        score_std = float(scenario_rows['score'].std()) if len(scenario_scores) > 1 else 0.0
        perfect_count = sum(1 for s in scenario_scores if s >= 1.0)
        
        return {
            'has_nr_cls': True,
            'score_mean': score_mean,
            'score_std': score_std,
            'total': num_scenarios,
            'perfect_count': perfect_count,
            'metrics': {},  # interPlan doesn't use individual metric breakdown like NR-CLS
            'metric_type': 'interplan_benchmark'
        }
    except Exception as e:
        return {
            'has_nr_cls': False,
            'error': f'Error reading interPlan metric file: {str(e)}'
        }


def calculate_nr_cls_score_from_multiple_dirs(metrics_dirs):
    """
    Calculate NR-CLS score by aggregating metrics from multiple batch directories.
    This is used when simulations were run in batches and each batch created its own directory.
    
    Args:
        metrics_dirs: List of metrics directory paths to aggregate from
        
    Returns:
        Same format as calculate_nr_cls_score()
    """
    # Aggregate all scenario scores from all batch directories
    all_scenario_scores = {}  # Key: scenario_name, Value: dict of metrics
    
    multiple_metrics = [
        'no_ego_at_fault_collisions',
        'drivable_area_compliance',
        'ego_is_making_progress',
        'driving_direction_compliance'
    ]
    
    weighted_metrics = {
        'ego_progress_along_expert_route': 5.0,
        'time_to_collision_within_bound': 5.0,
        'speed_limit_compliance': 4.0,
        'ego_is_comfortable': 2.0
    }
    
    all_metrics = multiple_metrics + list(weighted_metrics.keys())
    missing_metrics = []
    invalid_metrics = []
    
    # Aggregate metrics from all batch directories
    for metrics_dir in metrics_dirs:
        if not os.path.exists(metrics_dir):
            continue
            
        for metric_name in all_metrics:
            metric_file = os.path.join(metrics_dir, f'{metric_name}.parquet')
            if not os.path.exists(metric_file):
                if metric_name not in missing_metrics:
                    missing_metrics.append(metric_name)
                continue
            
            try:
                df = pd.read_parquet(metric_file)
                if 'scenario_name' not in df.columns or 'metric_score' not in df.columns:
                    continue
                
                for _, row in df.iterrows():
                    scenario_name = row['scenario_name']
                    metric_val = row['metric_score']
                    
                    if pd.notna(metric_val) and isinstance(metric_val, (int, float)):
                        if scenario_name not in all_scenario_scores:
                            all_scenario_scores[scenario_name] = {}
                        all_scenario_scores[scenario_name][metric_name] = metric_val
            except Exception as e:
                if f"{metric_name} ({str(e)})" not in invalid_metrics:
                    invalid_metrics.append(f"{metric_name} ({str(e)})")
                continue
    
    # Check if we have all required metrics
    required_metrics = multiple_metrics + list(weighted_metrics.keys())
    found_metrics = set()
    for scenario_metrics in all_scenario_scores.values():
        found_metrics.update(scenario_metrics.keys())
    
    missing_required = set(required_metrics) - found_metrics
    if missing_required:
        return {
            'has_nr_cls': False,
            'missing_metrics': list(missing_required),
            'error': f'Missing required metrics: {", ".join(missing_required)}'
        }
    
    # Calculate per-scenario NR-CLS scores
    scenario_nr_cls_scores = []
    perfect_count = 0
    sum_weights = sum(weighted_metrics.values())
    
    for scenario_name, scenario_metrics in all_scenario_scores.items():
        # Check if all required metrics are present
        has_all_multiple = all(m in scenario_metrics for m in multiple_metrics)
        has_all_weighted = all(m in scenario_metrics for m in weighted_metrics.keys())
        
        if not (has_all_multiple and has_all_weighted):
            continue
        
        # Calculate multiple_factor
        multiple_factor = 1.0
        for metric_name in multiple_metrics:
            multiple_factor *= scenario_metrics[metric_name]
        
        # Calculate weighted_average_score
        weighted_sum = sum(
            scenario_metrics[metric_name] * weight
            for metric_name, weight in weighted_metrics.items()
        )
        weighted_average_score = weighted_sum / sum_weights
        
        # Calculate final score
        scenario_score = multiple_factor * weighted_average_score
        scenario_nr_cls_scores.append(scenario_score)
        
        # Check if perfect
        all_perfect = (
            all(scenario_metrics.get(m, 0.0) == 1.0 for m in multiple_metrics) and
            all(scenario_metrics.get(m, 0.0) == 1.0 for m in weighted_metrics.keys())
        )
        if all_perfect:
            perfect_count += 1
    
    if not scenario_nr_cls_scores:
        return {
            'has_nr_cls': False,
            'error': 'No valid scenario scores calculated'
        }
    
    # Calculate mean and std
    total_score = sum(scenario_nr_cls_scores) / len(scenario_nr_cls_scores)
    total_std = np.std(scenario_nr_cls_scores) if len(scenario_nr_cls_scores) > 1 else 0.0
    
    # Calculate metric means for display
    scores = {}
    for metric_name in all_metrics:
        metric_vals = [scenario_metrics.get(metric_name) 
                      for scenario_metrics in all_scenario_scores.values()
                      if metric_name in scenario_metrics and pd.notna(scenario_metrics[metric_name])]
        
        if metric_vals:
            metric_type = 'multiple' if metric_name in multiple_metrics else 'weighted'
            weight = weighted_metrics.get(metric_name, None)
            mean_val = sum(metric_vals) / len(metric_vals)
            std_val = np.std(metric_vals) if len(metric_vals) > 1 else 0.0
            scores[metric_name] = {
                'mean': mean_val,
                'std': std_val,
                'count': len(metric_vals),
                'type': metric_type,
                'weight': weight,
            }
    
    return {
        'total': len(scenario_nr_cls_scores),
        'score_mean': total_score,
        'score_std': total_std,
        'perfect_count': perfect_count,
        'metrics': scores,
        'has_nr_cls': True,
        'missing_metrics': [],
        'invalid_metrics': [],
        'has_overwrites': False,
    }


def calculate_nr_cls_score(metrics_dir):
    """
    Calculate NR-CLS score from individual metric files using official nuplan-devkit formula.
    
    Formula (from closed_loop_nonreactive_agents_weighted_average.yaml):
    - multiple_factor = no_ego_at_fault_collisions × drivable_area_compliance × 
                        ego_is_making_progress × driving_direction_compliance
    - weighted_average_score = (5.0 * ego_progress_along_expert_route + 
                                5.0 * time_to_collision_within_bound + 
                                4.0 * speed_limit_compliance + 
                                2.0 * ego_is_comfortable) / (5.0 + 5.0 + 4.0 + 2.0)
    - final_score = multiple_factor × weighted_average_score
    
    This function now also checks for batch subdirectories and aggregates them.
    """
    
    # Multiple metrics (multiplied together)
    multiple_metrics = [
        'no_ego_at_fault_collisions',
        'drivable_area_compliance',
        'ego_is_making_progress',
        'driving_direction_compliance'
    ]
    
    # Weighted metrics (weighted average)
    weighted_metrics = {
        'ego_progress_along_expert_route': 5.0,
        'time_to_collision_within_bound': 5.0,
        'speed_limit_compliance': 4.0,
        'ego_is_comfortable': 2.0
    }
    
    all_metrics = multiple_metrics + list(weighted_metrics.keys())
    
    scores = {}
    scenario_scores_dict = {}  # Key: scenario_name, Value: dict of metrics
    missing_metrics = []
    invalid_metrics = []
    
    # First pass: Load all metrics and store by scenario_name
    for metric_name in all_metrics:
        metric_file = os.path.join(metrics_dir, f'{metric_name}.parquet')
        
        if not os.path.exists(metric_file):
            missing_metrics.append(metric_name)
            continue
        
        try:
            df = pd.read_parquet(metric_file)
            
            # Check if metric_score column exists
            if 'metric_score' not in df.columns:
                invalid_metrics.append(f"{metric_name} (no 'metric_score' column)")
                continue
            
            # Check if scenario_name column exists
            if 'scenario_name' not in df.columns:
                invalid_metrics.append(f"{metric_name} (no 'scenario_name' column)")
                continue
            
            # Store metric values by scenario_name
            # NOTE: If same scenario appears multiple times, only the LAST value is kept (overwrite)
            # This means if you run multiple models with same experiment name, results will be overwritten
            duplicate_count = 0
            unique_scenario_values = {}  # Track unique scenario values for mean calculation
            for _, row in df.iterrows():
                scenario_name = row['scenario_name']
                metric_val = row['metric_score']
                
                if pd.notna(metric_val) and isinstance(metric_val, (int, float)):
                    if scenario_name in scenario_scores_dict and metric_name in scenario_scores_dict[scenario_name]:
                        # Same scenario already exists - this will overwrite
                        duplicate_count += 1
                    if scenario_name not in scenario_scores_dict:
                        scenario_scores_dict[scenario_name] = {}
                    scenario_scores_dict[scenario_name][metric_name] = metric_val
                    # Store for mean calculation (will overwrite if duplicate, matching final score calculation)
                    unique_scenario_values[scenario_name] = metric_val
            
            if duplicate_count > 0:
                # This is just for logging - we'll warn at the end
                pass
            
            # Calculate metric mean from unique scenarios only (matching how final score is calculated)
            # This ensures the displayed means match what's actually used in the calculation
            valid_vals = list(unique_scenario_values.values())
            valid_vals = [v for v in valid_vals if pd.notna(v) and isinstance(v, (int, float))]
            
            if not valid_vals:
                invalid_metrics.append(f"{metric_name} (no valid numeric values)")
                continue
            
            # Store metric info with mean and std
            metric_type = 'multiple' if metric_name in multiple_metrics else 'weighted'
            weight = weighted_metrics.get(metric_name, None)
            mean_val = sum(valid_vals) / len(valid_vals)
            std_val = np.std(valid_vals) if len(valid_vals) > 1 else 0.0
            scores[metric_name] = {
                'mean': mean_val,
                'std': std_val,
                'count': len(valid_vals),
                'type': metric_type,
                'weight': weight,
                'values': valid_vals  # Store for later use if needed
            }
        except Exception as e:
            invalid_metrics.append(f"{metric_name} (error: {str(e)})")
            continue
    
    # Return detailed error if metrics are missing or invalid
    if missing_metrics or invalid_metrics:
        return {
            'has_nr_cls': False,
            'missing_metrics': missing_metrics,
            'invalid_metrics': invalid_metrics,
            'error': 'NR-CLS metrics not properly collected'
        }
    
    # Check if we have all required metrics
    required_metrics = multiple_metrics + list(weighted_metrics.keys())
    found_metrics = set(scores.keys())
    missing_required = set(required_metrics) - found_metrics
    
    if missing_required:
        return {
            'has_nr_cls': False,
            'missing_metrics': list(missing_required),
            'error': f'Missing required metrics: {", ".join(missing_required)}'
        }
    
    # Check for duplicate scenarios within the same metric file
    # (This would indicate multiple runs with same experiment name overwriting results)
    unique_scenarios = len(scenario_scores_dict)
    total_scenario_entries = 0
    for metric_name in all_metrics:
        metric_file = os.path.join(metrics_dir, f'{metric_name}.parquet')
        if os.path.exists(metric_file):
            df = pd.read_parquet(metric_file)
            if 'scenario_name' in df.columns:
                total_scenario_entries += len(df)
    
    # Calculate per-scenario NR-CLS scores using official formula
    scenario_nr_cls_scores = []
    perfect_count = 0
    
    # Calculate sum of weights for weighted average
    sum_weights = sum(weighted_metrics.values())  # 5.0 + 5.0 + 4.0 + 2.0 = 16.0
    
    for scenario_name, scenario_metrics in scenario_scores_dict.items():
        # Check if all required metrics are present
        has_all_multiple = all(m in scenario_metrics for m in multiple_metrics)
        has_all_weighted = all(m in scenario_metrics for m in weighted_metrics.keys())
        
        if not (has_all_multiple and has_all_weighted):
            # Missing some metrics for this scenario, skip
            continue
        
        # Calculate multiple_factor (product of multiple metrics)
        multiple_factor = 1.0
        for metric_name in multiple_metrics:
            multiple_factor *= scenario_metrics[metric_name]
        
        # Calculate weighted_average_score
        weighted_sum = sum(
            scenario_metrics[metric_name] * weight
            for metric_name, weight in weighted_metrics.items()
        )
        weighted_average_score = weighted_sum / sum_weights
        
        # Calculate final score (official formula)
        scenario_score = multiple_factor * weighted_average_score
        scenario_nr_cls_scores.append(scenario_score)
        
        # Check if perfect (all metrics = 1.0)
        all_perfect = (
            all(scenario_metrics.get(m, 0.0) == 1.0 for m in multiple_metrics) and
            all(scenario_metrics.get(m, 0.0) == 1.0 for m in weighted_metrics.keys())
        )
        if all_perfect:
            perfect_count += 1
    
    if not scenario_nr_cls_scores:
        return {
            'has_nr_cls': False,
            'error': 'No valid scenario scores calculated'
        }
    
    # Calculate mean and std of per-scenario scores (this is the correct NR-CLS final score mean)
    total_score = sum(scenario_nr_cls_scores) / len(scenario_nr_cls_scores)
    total_std = np.std(scenario_nr_cls_scores) if len(scenario_nr_cls_scores) > 1 else 0.0
    total_scenarios = len(scenario_nr_cls_scores)
    
    # Check if there might be overwritten results
    # If total entries > unique scenarios * num_metrics, there might be duplicates
    expected_entries = unique_scenarios * len(all_metrics)
    has_overwrites = False
    if total_scenario_entries > expected_entries:
        has_overwrites = True
    
    return {
        'total': total_scenarios,
        'score_mean': total_score,
        'score_std': total_std,
        'perfect_count': perfect_count,
        'metrics': scores,
        'has_nr_cls': True,
        'missing_metrics': [],
        'invalid_metrics': [],
        'has_overwrites': has_overwrites,
        'total_entries': total_scenario_entries,
        'expected_entries': expected_entries
    }

def load_scenario_records(records_dir):
    """
    Load scenario token records from JSON files.
    
    Returns:
        dict: {experiment_name: set of scenario_tokens}
    """
    records = {}
    if not os.path.exists(records_dir):
        return records
    
    for json_file in Path(records_dir).glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                experiment_name = json_file.stem
                scenario_tokens = set(data.get('scenario_tokens', data.get('tokens', [])))
                records[experiment_name] = scenario_tokens
        except Exception as e:
            print(f"⚠️  Could not load record {json_file}: {e}")
    
    return records


def load_scenario_record_metadata(records_dir):
    """Load full scenario record metadata keyed by experiment name."""
    records = {}
    if not os.path.exists(records_dir):
        return records

    for json_file in Path(records_dir).glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                records[json_file.stem] = json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load record metadata {json_file}: {e}")

    return records

def main():
    base_dir = os.environ.get(
        "NUPLAN_EXP_DIR",
        str((REPO_ROOT / "../nuplan-devkit/nuplan/exp/exp").resolve()),
    )
    records_dir = os.environ.get(
        "PLUTO_SCENARIO_RECORDS_DIR",
        str(REPO_ROOT / "artifacts/records/scenario_records"),
    )
    
    # Load scenario token records
    scenario_records = load_scenario_records(records_dir)
    scenario_record_metadata = load_scenario_record_metadata(records_dir)
    if scenario_records:
        total_recorded = sum(len(tokens) for tokens in scenario_records.values())
        print(f"📝 Loaded scenario records: {len(scenario_records)} experiments, {total_recorded} total scenarios")
        print(f"   Records directory: {records_dir}")
    
    test_groups = [
        ('Val14', 'val14_benchmark'),
        ('Test14-hard', 'test14_hard'), # ('Val14', 'val14_benchmark_half'),
        ('InterPlan (interplan10)', 'interplan10'),
        ('InterPlan (benchmark_scenarios)', 'benchmark_scenarios'),
    ] 
    
    print("="*80)
    print("QUICK TEST RESULTS ANALYSIS: HEAD-ONLY FINE-TUNING")
    print("Target: 70 scenarios per group (actual may vary due to nuplan_mini DB)")
    print("Metric Validation (NR-CLS for nuPlan, interPlan Benchmark for interPlan)")
    print("="*80)

    all_results = []
    has_valid_data = False
    failed_tests = []
    metric_validation_passed = True
    
    for group_name, suffix in test_groups:
        # Set target scenario count based on test group
        if 'InterPlan' in group_name:
            if 'interplan10' in suffix:
                target_scenarios = 80  # InterPlan interplan10: 8 types × 10 per type
            elif 'benchmark_scenarios' in suffix:
                target_scenarios = 335  # InterPlan benchmark_scenarios: all scenarios
            else:
                target_scenarios = 80  # Default for interPlan
        else:
            target_scenarios = 70  # Default for other tests
        
        print(f"\n{'='*80}")
        print(f"{group_name} (Target: {target_scenarios} scenarios)")
        print(f"{'='*80}")
        
        # Note: zero-shot may be commented out in quick_test.sh.
        # Include current explicit curriculum slugs plus legacy slugs for old runs.
        if 'InterPlan' in group_name:
            methods = ['zeroshot', 'rulebased', 'lossbased', 'curriculum_uniform', 'curriculum_llmbased', 'uniform', 'curriculum']
        else:
            methods = ['zeroshot', 'rulebased', 'lossbased', 'curriculum_uniform', 'curriculum_llmbased']

        for method in methods:
            # Handle both regular and interplan experiments
            if 'interplan10' in suffix or 'benchmark_scenarios' in suffix:
                # interPlan experiments use format: quick_test_interplan_{method}_{filter}
                exp_dir = os.path.join(base_dir, f'quick_test_interplan_{method}_{suffix}')
                exp_name = f'quick_test_interplan_{method}_{suffix}'
                # interPlan uses default_interplan_benchmark subdirectory with timestamp subdirectories
                interplan_benchmark_dir = os.path.join(exp_dir, 'default_interplan_benchmark')
                if os.path.exists(interplan_benchmark_dir):
                    # Find the most recent timestamp directory (format: YYYY.MM.DD.HH.MM.SS)
                    import re
                    timestamp_dirs = [d for d in os.listdir(interplan_benchmark_dir) 
                                    if os.path.isdir(os.path.join(interplan_benchmark_dir, d)) and 
                                    re.match(r'\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}', d)]
                    if timestamp_dirs:
                        # Sort by timestamp (directory name format: YYYY.MM.DD.HH.MM.SS)
                        timestamp_dirs.sort(reverse=True)
                        most_recent_dir = os.path.join(interplan_benchmark_dir, timestamp_dirs[0])
                        # interPlan stores aggregated metrics in aggregator_metric subdirectory
                        aggregator_metric_dir = os.path.join(most_recent_dir, 'aggregator_metric')
                        base_metrics_dir = aggregator_metric_dir if os.path.exists(aggregator_metric_dir) else os.path.join(most_recent_dir, 'metrics')
                    else:
                        base_metrics_dir = os.path.join(interplan_benchmark_dir, 'metrics') # Fallback if no timestamp dir
                else:
                    base_metrics_dir = os.path.join(exp_dir, 'metrics') # Fallback if no interplan_benchmark_dir
            else:
                exp_dir = os.path.join(base_dir, f'quick_test_{method}_{suffix}')
                base_metrics_dir = os.path.join(exp_dir, 'metrics')
                exp_name = f'quick_test_{method}_{suffix}'
            
            print(f"\n📋 {method.upper()}:")
            
            # Check for batch subdirectories (from batched runs)
            batch_dirs = find_batch_directories(exp_dir)
            metrics_dirs = []
            
            # First, check batch directories (new batched runs with unique experiment names)
            if batch_dirs:
                print(f"   📦 Found {len(batch_dirs)} batch subdirectories")
                for batch_dir in batch_dirs:
                    batch_metrics_dir = os.path.join(batch_dir, 'metrics')
                    if os.path.exists(batch_metrics_dir):
                        # Check if metrics directory has any parquet files
                        try:
                            metric_files = [f for f in os.listdir(batch_metrics_dir) if f.endswith('.parquet')]
                            if metric_files:
                                metrics_dirs.append(batch_metrics_dir)
                        except:
                            pass
            
            # Also check base directory (for non-batched runs or if batches wrote to same dir)
            if os.path.exists(base_metrics_dir):
                try:
                    metric_files = [f for f in os.listdir(base_metrics_dir) if f.endswith('.parquet')]
                    if metric_files:
                        if base_metrics_dir not in metrics_dirs:
                            metrics_dirs.append(base_metrics_dir)
                except:
                    pass

            record_metadata = scenario_record_metadata.get(exp_name, {})
            resolved_from_record = []
            for raw_path in record_metadata.get('resolved_metrics_dirs', []):
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = (REPO_ROOT / candidate).resolve()
                if candidate.exists():
                    resolved_from_record.append(str(candidate))
            if resolved_from_record:
                metrics_dirs = resolved_from_record
                print(f"   📝 Using {len(metrics_dirs)} metrics dir(s) from scenario record metadata")
            
            if not metrics_dirs:
                print(f"   ❌ Metrics directory not found: {base_metrics_dir}")
                if batch_dirs:
                    print(f"      (Found {len(batch_dirs)} batch directories but no metrics in them)")
                failed_tests.append((group_name, method, 'Metrics directory not found'))
                metric_validation_passed = False
                continue
            
            # Calculate aggregated score from all metrics directories
            # For interPlan, use interPlan's own metric aggregator file
            if 'interplan10' in suffix or 'benchmark_scenarios' in suffix:
                # interPlan stores results in aggregator_metric directory
                result = calculate_interplan_score(metrics_dirs[0] if metrics_dirs else base_metrics_dir)
            elif len(metrics_dirs) > 1:
                # Aggregate from multiple directories (batch directories + base)
                print(f"   📦 Aggregating metrics from {len(metrics_dirs)} directories (batches + base)...")
                result = calculate_nr_cls_score_from_multiple_dirs(metrics_dirs)
            else:
                # Single directory (non-batched or aggregated already)
                result = calculate_nr_cls_score(metrics_dirs[0])
            
            # Warn if results seem incomplete
            if result and result.get('has_nr_cls', False):
                total_scenarios = result.get('total', 0)
                if batch_dirs and total_scenarios < 100:  # Suspiciously low for batched runs
                    print(f"   ⚠️  WARNING: Only {total_scenarios} scenarios found, but {len(batch_dirs)} batch directories detected!")
                    print(f"      This might indicate batches overwrote each other's metrics in the base directory.")
                    print(f"      Future runs with updated batched script will create separate batch directories.")
                    print(f"      Consider re-running to get all scenarios from separate batch directories.")
            
            if result is None or not result.get('has_nr_cls', False):
                metric_type = result.get('metric_type', 'NR-CLS') if result else 'NR-CLS'
                print(f"   ❌ {metric_type} metrics validation FAILED")
                if result and 'missing_metrics' in result:
                    if result['missing_metrics']:
                        print(f"      Missing files: {', '.join(result['missing_metrics'])}")
                    if result['invalid_metrics']:
                        print(f"      Invalid metrics: {', '.join(result['invalid_metrics'])}")
                    if 'error' in result:
                        print(f"      Error: {result['error']}")
                failed_tests.append((group_name, method, f'{metric_type} metrics invalid'))
                metric_validation_passed = False
                continue
            
            # Metrics are valid!
            result['method'] = method
            result['group'] = group_name
            all_results.append(result)
            has_valid_data = True
            
            metric_type = result.get('metric_type', 'NR-CLS')
            metric_label = 'interPlan Benchmark Score' if metric_type == 'interplan_benchmark' else 'NR-CLS Final Score'
            print(f"   ✅ {metric_type} metrics: VALID")
            # Get target scenarios for this group
            if 'InterPlan' in group_name:
                if 'interplan10' in suffix:
                    target_scenarios = 80
                elif 'benchmark_scenarios' in suffix:
                    target_scenarios = 335
                else:
                    target_scenarios = 80
            else:
                target_scenarios = 70
            print(f"   📊 Scenarios: {result['total']} (Target: {target_scenarios}, actual may vary)")
            score_std = result.get('score_std', 0.0)
            print(f"   📊 {metric_label}: {result['score_mean']:.4f} ± {score_std:.4f}")
            print(f"   📊 Perfect: {result['perfect_count']}/{result['total']} ({result['perfect_count']/result['total']*100:.1f}%)")
            
            # Show if scenario tokens are recorded
            if exp_name in scenario_records:
                recorded_count = len(scenario_records[exp_name])
                print(f"   📝 Recorded scenarios: {recorded_count} (saved to artifacts/records/scenario_records/{exp_name}.json)")
            
            print(f"   📝 Note: Final score = mean(per-scenario scores), where each scenario score =")
            print(f"      (multiple_factor) × (weighted_average), calculated per scenario")
            
            # Warn if results might be overwritten
            if result.get('has_overwrites', False):
                print(f"   ⚠️  WARNING: Possible overwritten results detected!")
                print(f"      Total entries: {result.get('total_entries', 'N/A')}, Expected: {result.get('expected_entries', 'N/A')}")
                print(f"      If you ran multiple models with same experiment name, only LAST run is used")
            
            print(f"\n   Individual Metrics (mean ± std across unique scenarios):")
            # Show multiple metrics (multiplied)
            print(f"      Multiple Metrics (multiplied):")
            for metric_name, metric_data in result['metrics'].items():
                if metric_data.get('type') == 'multiple':
                    metric_short = metric_name.replace('_', ' ').title()
                    status_icon = "✅" if metric_data['mean'] == 1.0 else "⚠️"
                    metric_std = metric_data.get('std', 0.0)
                    print(f"         {status_icon} {metric_short}: {metric_data['mean']:.4f} ± {metric_std:.4f}")
            # Show weighted metrics (weighted average)
            print(f"      Weighted Metrics (weighted average):")
            for metric_name, metric_data in result['metrics'].items():
                if metric_data.get('type') == 'weighted':
                    metric_short = metric_name.replace('_', ' ').title()
                    status_icon = "✅" if metric_data['mean'] == 1.0 else "⚠️"
                    weight = metric_data.get('weight', 'N/A')
                    metric_std = metric_data.get('std', 0.0)
                    print(f"         {status_icon} {metric_short}: {metric_data['mean']:.4f} ± {metric_std:.4f} (weight: {weight})")
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    # First check: NR-CLS metric validation
    print("\n" + "="*80)
    print("NR-CLS METRIC VALIDATION")
    print("="*80)
    
    if not metric_validation_passed:
        print("\n❌ NR-CLS METRIC VALIDATION FAILED!")
        print("   Some tests did not produce valid NR-CLS metrics.")
        print("   Check simulation configuration and metric collection.")
        if failed_tests:
            print(f"\n   Failed tests ({len(failed_tests)}):")
            for group, method, reason in failed_tests:
                print(f"      • {group} - {method.upper()}: {reason}")
        return 1
    else:
        print("\n✅ NR-CLS METRIC VALIDATION PASSED!")
        print("   All tests produced valid NR-CLS metrics.")
        print("   All 8 required metrics are present:")
        print("     - Multiple metrics (4): collisions, drivable_area, direction, progress")
        print("     - Weighted metrics (4): ego_progress, ttc, speed_limit, comfort")
    
    if not all_results:
        print("\n❌ NO DATA FOUND!")
        print("   Check if simulations ran successfully")
        return 1
    
    if has_valid_data:
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        print("\n📊 Comparison Table:")
        print(f"\n{'Group':<15} {'Method':<12} {'Count':<8} {'Score (mean±std)':<20} {'Perfect%':<10}")
        print("-" * 75)
        
        for result in all_results:
            score_std = result.get('score_std', 0.0)
            score_str = f"{result['score_mean']:.4f}±{score_std:.4f}"
            perfect_pct = result['perfect_count']/result['total']*100
            perfect_str = f"{perfect_pct:.1f}%"
            
            print(f"{result['group']:<15} {result['method']:<12} {result['total']:<8} {score_str:<20} {perfect_str:<10}")
        
        # Compare all 3 methods
        print("\n" + "="*80)
        print("COMPARISON: HEAD-ONLY FINE-TUNING (Uniform vs Curriculum)")
        print("="*80)
        print("Note: Zero-shot may be commented out in quick_test.sh")
        
        groups_dict = {}
        for result in all_results:
            if result['group'] not in groups_dict:
                groups_dict[result['group']] = {}
            groups_dict[result['group']][result['method']] = result
        
        for group_name, suffix in test_groups:
            if group_name not in groups_dict:
                continue
            
            print(f"\n{group_name}:")
            group_data = groups_dict[group_name]
            
            if 'zeroshot' in group_data:
                z = group_data['zeroshot']
                z_std = z.get('score_std', 0.0)
                print(f"   Zero-shot:  Score {z['score_mean']:.4f} ± {z_std:.4f}, Perfect {z['perfect_count']}/{z['total']} ({z['perfect_count']/z['total']*100:.1f}%)")
            
            if 'uniform' in group_data:
                u = group_data['uniform']
                u_std = u.get('score_std', 0.0)
                print(f"   Uniform:    Score {u['score_mean']:.4f} ± {u_std:.4f}, Perfect {u['perfect_count']}/{u['total']} ({u['perfect_count']/u['total']*100:.1f}%)")
            
            if 'curriculum' in group_data:
                c = group_data['curriculum']
                c_std = c.get('score_std', 0.0)
                print(f"   Curriculum: Score {c['score_mean']:.4f} ± {c_std:.4f}, Perfect {c['perfect_count']}/{c['total']} ({c['perfect_count']/c['total']*100:.1f}%)")
            
            # Find best
            if 'zeroshot' in group_data and 'uniform' in group_data and 'curriculum' in group_data:
                z_score = group_data['zeroshot']['score_mean']
                u_score = group_data['uniform']['score_mean']
                c_score = group_data['curriculum']['score_mean']
                
                best_score = max(z_score, u_score, c_score)
                if best_score == c_score:
                    c_std = group_data['curriculum'].get('score_std', 0.0)
                    print(f"   ✅ Best: Curriculum ({c_score:.4f} ± {c_std:.4f})")
                elif best_score == u_score:
                    u_std = group_data['uniform'].get('score_std', 0.0)
                    print(f"   ✅ Best: Uniform ({u_score:.4f} ± {u_std:.4f})")
                elif best_score == z_score:
                    z_std = group_data['zeroshot'].get('score_std', 0.0)
                    print(f"   ✅ Best: Zero-shot ({z_score:.4f} ± {z_std:.4f})")
        
        print("\n" + "="*80)
        print("FINAL STATUS")
        print("="*80)
        print("\n✅ NR-CLS METRICS VALID - Head-only fine-tuning results ready!")
        print("   • NR-CLS metrics: ✅ Valid")
        print("   • Methods tested: ✅ Uniform (head-only), Curriculum (head-only)")
        print("   • Note: Head-only fine-tuning trains only planning decoder final heads")
        
        # Show scenario records summary
        if scenario_records:
            print(f"\n📝 Scenario Records:")
            print(f"   • Total experiments recorded: {len(scenario_records)}")
            total_scenarios = sum(len(tokens) for tokens in scenario_records.values())
            print(f"   • Total scenarios recorded: {total_scenarios}")
            print(f"   • Records location: {records_dir}")
            print(f"   • Use these records to skip already-run scenarios in future tests")
        
        print("\nRun full tests with:")
        print("  bash test.sh              # Current consolidated benchmark runner")
        print("  Archived older variants are under archive/legacy_scripts/tests/")
        
        return 0
    else:
        print("\n❌ VALIDATION FAILED - No valid metrics found!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
