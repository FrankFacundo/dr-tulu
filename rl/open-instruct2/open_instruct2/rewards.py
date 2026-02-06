from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from open_instruct2.bootstrap import ensure_open_instruct_importable
from open_instruct2.config import TrainArgs

ensure_open_instruct_importable()

from open_instruct.ground_truth_utils import (  # noqa: E402
    build_all_verifiers,
    is_a_good_rl_rag_response,
    soft_format_reward_func,
)
from open_instruct.model_utils import apply_verifiable_reward  # noqa: E402
from open_instruct.search_rewards.longform_rubric_rewards import (  # noqa: E402
    create_rubric_key,
)

LOGGER = logging.getLogger(__name__)


def _parse_ground_truth_obj(ground_truth: Union[str, List[str]]) -> Dict:
    if isinstance(ground_truth, list):
        if not ground_truth:
            return {}
        ground_truth = ground_truth[0]
    if isinstance(ground_truth, str):
        return json.loads(ground_truth)
    raise ValueError(f"Unsupported ground truth type: {type(ground_truth)}")


class RewardEngine:
    def __init__(self, args: TrainArgs) -> None:
        self.args = args
        self.reward_fn_mapping = build_all_verifiers(args) if args.apply_verifiable_reward else {}

    def initialize_rubric_buffer(self, train_ground_truths: Sequence[Union[str, List[str]]]) -> Optional[Dict]:
        if not (self.args.apply_adaptive_rubric_reward and self.args.use_rubric_buffer):
            return None

        rubric_buffer: Dict = {}
        for gt in train_ground_truths:
            try:
                gt_obj = _parse_ground_truth_obj(gt)
            except Exception as exc:
                LOGGER.warning("Skipping invalid ground truth for rubric buffer init: %s", exc)
                continue

            query = gt_obj.get("query")
            rubrics = gt_obj.get("rubrics", [])
            if not query:
                continue
            rubric_buffer[query] = {
                "active_rubrics": [] if self.args.use_static_rubrics_as_persistent_rubrics else list(rubrics),
                "inactive_rubrics": [],
                "persistent_rubrics": list(rubrics) if self.args.use_static_rubrics_as_persistent_rubrics else [],
                "static_rubrics": list(rubrics),
            }
        LOGGER.info("Initialized rubric buffer with %d queries.", len(rubric_buffer))
        return rubric_buffer

    def maybe_add_static_rubrics(self, rubric_buffer: Optional[Dict], training_step: int) -> int:
        if rubric_buffer is None:
            return 0
        if self.args.use_static_rubrics_as_persistent_rubrics:
            return 0
        if training_step % self.args.add_static_rubrics_to_active_rubrics_every_n_steps != 0:
            return 0

        added_count = 0
        for _, buffer_data in rubric_buffer.items():
            static_rubrics = buffer_data.get("static_rubrics", [])
            active_rubrics = buffer_data.get("active_rubrics", [])
            for rubric in static_rubrics:
                if rubric not in active_rubrics:
                    active_rubrics.append(rubric)
                    added_count += 1
        return added_count

    async def compute(
        self,
        responses: List[torch.Tensor],
        decoded_responses: List[str],
        ground_truths: List[Union[str, List[str]]],
        datasets: List[Union[str, List[str]]],
        finish_reasons: List[str],
        queries: Optional[List[str]] = None,
        source_datasets: Optional[List[str]] = None,
        rubric_buffer: Optional[Dict] = None,
        is_training: bool = True,
        training_step: Optional[int] = None,
    ) -> Tuple[List[float], Dict[str, float], Optional[Dict], Optional[List[Dict]]]:
        if queries is None:
            queries = [None] * len(responses)

        scores = [0.0] * len(decoded_responses)
        metrics: Dict[str, float] = {}
        log_values: Dict[str, List] = {}
        adaptive_rubric_scores_for_saving: Optional[List[Dict]] = None

        if self.args.apply_r1_style_format_reward:
            format_scores = soft_format_reward_func(decoded_responses, self.args.r1_style_format_reward)
            for i in range(len(scores)):
                scores[i] += format_scores[i]
            metrics["val/format_scores"] = float(np.mean(format_scores))
        elif self.args.apply_rl_rag_format_reward:
            format_scores = is_a_good_rl_rag_response(decoded_responses)
            for i in range(len(scores)):
                scores[i] += format_scores[i]
            metrics["val/rl_rag_format_scores"] = float(np.mean(format_scores))
        else:
            format_scores = None

        if self.args.apply_adaptive_rubric_reward and is_training:
            from open_instruct.search_rewards.utils.rubric_utils import (  # noqa: WPS433,E402
                _generate_instance_wise_adaptive_rubrics,
                save_adaptive_rubric_cache_safe,
                update_ground_truths_with_adaptive_rubrics,
            )

            all_adaptive_rubrics, num_subsampled_answers_list = await _generate_instance_wise_adaptive_rubrics(
                decoded_responses,
                ground_truths,
                self.args.num_samples_per_prompt_rollout,
                rubric_buffer=rubric_buffer,
                use_full_responses=self.args.use_full_responses_for_adaptive_rubric,
                answer_length_limit_in_words=self.args.answer_length_limit_in_words,
            )
            adaptive_rubric_scores_for_saving = all_adaptive_rubrics
            if self.args.cache_adaptive_rubric_data_dir and training_step is not None:
                try:
                    save_adaptive_rubric_cache_safe(
                        cache_dir=self.args.cache_adaptive_rubric_data_dir,
                        training_step=training_step,
                        decoded_responses=decoded_responses,
                        ground_truths=ground_truths,
                        all_adaptive_rubrics=all_adaptive_rubrics,
                        num_subsampled_answers_list=num_subsampled_answers_list,
                        num_samples_per_prompt_rollout=self.args.num_samples_per_prompt_rollout,
                        use_full_responses=self.args.use_full_responses_for_adaptive_rubric,
                        answer_length_limit_in_words=self.args.answer_length_limit_in_words,
                    )
                except Exception as exc:
                    LOGGER.warning("Failed to cache adaptive rubric data at step %s: %s", training_step, exc)

            (
                ground_truths,
                valid_rate,
                avg_num_ground_truths,
                avg_num_adaptive_rubrics,
                avg_num_active_buffer_rubrics,
                rubric_buffer,
                skipped_count,
            ) = update_ground_truths_with_adaptive_rubrics(
                ground_truths,
                all_adaptive_rubrics,
                self.args.num_samples_per_prompt_rollout,
                rubric_buffer=rubric_buffer,
            )
            metrics["objective/valid_adaptive_rubric_rate"] = float(valid_rate)
            metrics["objective/avg_num_ground_truths"] = float(avg_num_ground_truths)
            metrics["objective/avg_num_adaptive_rubrics"] = float(avg_num_adaptive_rubrics)
            metrics["objective/avg_num_active_buffer_rubrics"] = float(avg_num_active_buffer_rubrics)
            metrics["objective/skipped_adaptive_rubrics"] = float(skipped_count)

        if self.args.apply_verifiable_reward:
            verifiable_rewards, per_func_rewards, log_values = await apply_verifiable_reward(
                self.reward_fn_mapping,
                responses,
                decoded_responses,
                ground_truths,
                datasets,
                reward_mult=self.args.verification_reward,
                queries=queries,
                overwrite_reward_fn_tag=self.args.overwrite_reward_fn_tag,
            )
            if len(verifiable_rewards) != len(scores):
                raise ValueError(f"{len(verifiable_rewards)=} != {len(scores)=}")

            for i in range(len(verifiable_rewards)):
                if self.args.apply_r1_style_format_reward or self.args.apply_rl_rag_format_reward:
                    if self.args.additive_format_reward:
                        scores[i] = float(verifiable_rewards[i]) + scores[i]
                    else:
                        assert format_scores is not None
                        scores[i] = float(verifiable_rewards[i]) if format_scores[i] == 1 else 0.0
                else:
                    scores[i] = float(verifiable_rewards[i])

            np_verifiable_rewards = np.array(verifiable_rewards, dtype=np.float32)
            metrics["objective/verifiable_reward"] = float(np_verifiable_rewards.mean())
            metrics["objective/verifiable_correct_rate"] = float((np_verifiable_rewards > 0.0).mean())

            for key, value in log_values.items():
                if key in {"rubric_scores_by_title", "per_rubric_rewards"}:
                    continue
                metrics[f"objective/reward_log_values/{key}"] = float(np.array(value).mean())

            per_func_lists: Dict[str, List[float]] = defaultdict(list)
            for reward_dict in per_func_rewards:
                for key, value in reward_dict.items():
                    per_func_lists[key].append(value)
            for key, value in per_func_lists.items():
                arr = np.array(value, dtype=np.float32)
                metrics[f"objective/{key}_reward"] = float(arr.mean())
                metrics[f"objective/{key}_correct_rate"] = float((arr > 0.0).mean())

            if source_datasets is not None and len(source_datasets) == len(verifiable_rewards):
                source_to_values: Dict[str, List[float]] = defaultdict(list)
                for src, val in zip(source_datasets, verifiable_rewards):
                    source_to_values[str(src)].append(float(val))
                for src, vals in source_to_values.items():
                    arr = np.array(vals, dtype=np.float32)
                    metrics[f"objective/source/{src}_verifiable_reward"] = float(arr.mean())
                    metrics[f"objective/source/{src}_verifiable_correct_rate"] = float((arr > 0.0).mean())

            if self.args.log_direction_agreement and log_values:
                try:
                    from open_instruct.search_rewards.utils._direction_agreement import (  # noqa: WPS433,E402
                        compute_direction_agreement,
                    )

                    direction = compute_direction_agreement(log_values, verifiable_rewards)
                    for key, val in direction.items():
                        metrics[f"analysis/direction_agreement/{key}"] = float(val)
                except Exception as exc:
                    LOGGER.warning("Direction agreement logging failed: %s", exc)

        if self.args.non_stop_penalty:
            for i, reason in enumerate(finish_reasons):
                if reason != "stop":
                    scores[i] = self.args.non_stop_penalty_value

        rubric_key_stats = None
        if (
            is_training
            and self.args.normalize_rubric_scores
            and isinstance(log_values, dict)
            and "rubric_scores_by_title" in log_values
        ):
            scores, norm_metrics, rubric_key_stats = self._normalize_rubric_scores(
                scores=scores,
                log_values=log_values,
                ground_truths=ground_truths,
            )
            metrics.update(norm_metrics)

            if self.args.apply_adaptive_rubric_reward and rubric_buffer is not None and rubric_key_stats is not None:
                moved_zero_std, moved_low_std = self._filter_rubric_buffer(rubric_buffer, rubric_key_stats)
                if moved_zero_std or moved_low_std:
                    metrics["rubric_keys/moved_zero_std_to_inactive"] = float(moved_zero_std)
                    metrics["rubric_keys/moved_low_std_to_inactive"] = float(moved_low_std)

        return scores, metrics, rubric_buffer, adaptive_rubric_scores_for_saving

    def _normalize_rubric_scores(
        self,
        scores: List[float],
        log_values: Dict[str, List],
        ground_truths: List[Union[str, List[str]]],
    ) -> Tuple[List[float], Dict[str, float], Optional[Dict[str, Dict[str, float]]]]:
        rubric_scores_by_title = log_values.get("rubric_scores_by_title", [])
        if not rubric_scores_by_title:
            return scores, {}, None

        limit = min(len(scores), len(rubric_scores_by_title), len(ground_truths))
        if limit == 0:
            return scores, {}, None

        rubric_key_scores: Dict[str, List[float]] = defaultdict(list)
        for response_scores in rubric_scores_by_title[:limit]:
            if not isinstance(response_scores, dict):
                continue
            for rubric_key, score in response_scores.items():
                rubric_key_scores[rubric_key].append(float(score))

        rubric_key_stats: Dict[str, Dict[str, float]] = {}
        for rubric_key, values in rubric_key_scores.items():
            arr = np.array(values, dtype=np.float32)
            rubric_key_stats[rubric_key] = {"mean": float(arr.mean()), "std": float(arr.std())}

        normalized_scores = list(scores)
        final_advantages = np.zeros(limit, dtype=np.float32)
        original_scores = np.array(scores[:limit], dtype=np.float32)

        for i in range(limit):
            response_scores = rubric_scores_by_title[i]
            if not isinstance(response_scores, dict):
                continue
            gt_obj = _parse_ground_truth_obj(ground_truths[i])
            query = gt_obj.get("query", "")
            rubrics = gt_obj.get("rubrics", [])
            rubric_key_weights: Dict[str, List[float]] = defaultdict(list)
            for rubric in rubrics:
                rubric_key = create_rubric_key(query, rubric)
                rubric_key_weights[rubric_key].append(float(rubric.get("weight", 1.0)))
            weight_map = {
                rubric_key: float(np.mean(weight_values))
                for rubric_key, weight_values in rubric_key_weights.items()
            }

            weighted_sum = 0.0
            total_weight = 0.0
            for rubric_key, raw_score in response_scores.items():
                stats = rubric_key_stats.get(rubric_key)
                if stats is None:
                    continue
                std = stats["std"]
                normalized = (float(raw_score) - stats["mean"]) / std if std > 0 else 0.0
                weight = float(weight_map.get(rubric_key, 1.0))
                weighted_sum += normalized * weight
                if weight > 0:
                    total_weight += weight
            normalized = weighted_sum / max(total_weight, 1.0)
            final_advantages[i] = normalized
            normalized_scores[i] = normalized

        metrics: Dict[str, float] = {}
        if rubric_key_stats:
            means = np.array([stats["mean"] for stats in rubric_key_stats.values()], dtype=np.float32)
            stds = np.array([stats["std"] for stats in rubric_key_stats.values()], dtype=np.float32)
            metrics["rubric_keys/avg_mean"] = float(means.mean())
            metrics["rubric_keys/avg_std"] = float(stds.mean())
            metrics["rubric_keys/num_all_zero_rubrics_ratio"] = float(((means == 0) & (stds == 0)).mean())
            metrics["rubric_keys/num_all_same_value_non_zero_rubrics_ratio"] = float(((means > 0) & (stds == 0)).mean())
            if len(final_advantages) > 1:
                corr = np.corrcoef(final_advantages, original_scores)[0, 1]
                if np.isfinite(corr):
                    metrics["rubric_keys/advantage_corr"] = float(corr)
        metrics["rubric_keys/num_responses_with_non_zero_advantage"] = float((final_advantages != 0).sum())
        return normalized_scores, metrics, rubric_key_stats

    def _filter_rubric_buffer(self, rubric_buffer: Dict, rubric_key_stats: Dict[str, Dict[str, float]]) -> Tuple[int, int]:
        moved_zero_std = 0
        moved_low_std = 0

        for query, buffer_data in rubric_buffer.items():
            active_rubrics = list(buffer_data.get("active_rubrics", []))
            inactive_rubrics = buffer_data.get("inactive_rubrics", [])

            kept_rubrics: List[Dict] = []
            rubric_std_pairs: List[Tuple[Dict, float]] = []

            for rubric in active_rubrics:
                rubric_key = create_rubric_key(query, rubric)
                stats = rubric_key_stats.get(rubric_key)
                if stats is None:
                    kept_rubrics.append(rubric)
                    continue
                if stats["std"] == 0:
                    inactive_rubrics.append(rubric)
                    moved_zero_std += 1
                else:
                    kept_rubrics.append(rubric)
                    rubric_std_pairs.append((rubric, stats["std"]))

            if len(kept_rubrics) > self.args.max_active_rubrics:
                rubric_std_pairs.sort(key=lambda item: item[1], reverse=True)
                keep_keys = {
                    create_rubric_key(query, rubric)
                    for rubric, _ in rubric_std_pairs[: self.args.max_active_rubrics]
                }
                capped_rubrics: List[Dict] = []
                for rubric in kept_rubrics:
                    rubric_key = create_rubric_key(query, rubric)
                    if rubric_key in keep_keys or rubric_key not in rubric_key_stats:
                        capped_rubrics.append(rubric)
                    else:
                        inactive_rubrics.append(rubric)
                        moved_low_std += 1
                kept_rubrics = capped_rubrics

            buffer_data["active_rubrics"] = kept_rubrics
            buffer_data["inactive_rubrics"] = inactive_rubrics

        return moved_zero_std, moved_low_std
