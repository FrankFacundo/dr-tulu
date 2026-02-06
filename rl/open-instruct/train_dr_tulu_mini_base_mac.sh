model_path=Qwen/Qwen3-0.6B
dataset_name="rl-research/dr-tulu-rl-data"
dataset_list="${dataset_name} 1.0"
exp_name="dr-tulu-mini-base-mac"

export PYTORCH_ENABLE_MPS_FALLBACK=1

uv run python open_instruct/grpo_fast.py \
        --exp_name ${exp_name} \
        --training_backend torch \
        --rollout_backend hf \
        --device mps \
        --num_learners_per_node 1 \
        --single_gpu_mode True \
        --num_samples_per_prompt_rollout 2 \
        --num_unique_prompts_rollout 4 \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --output_dir output \
        --kl_estimator kl3 \
        --dataset_mixer_list ${dataset_list} \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list ${dataset_name} 16 \
        --dataset_mixer_eval_list_splits train \
        --max_token_length 2048 \
        --max_prompt_token_length 512 \
        --response_length 512 \
        --pack_length 1024 \
        --model_name_or_path ${model_path} \
        --total_episodes 128 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --num_evals 4 \
        --save_freq -1 \
        --try_launch_beaker_eval_jobs_on_weka False
