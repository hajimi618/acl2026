# 4*80G GPU

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model /DeepSeek-R1-7B-CoT-SFT \
    --train_type full \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 4096 \
    --vllm_tensor_parallel_size 4 \
    --dataset /Data/GRPO-CoT-Data-1 \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --max_length 4096 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --learning_rate 1e-6 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --output_dir /Save \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_completion_length 2048 \
    --reward_funcs chem_reward \
    --num_generations 4 \
    --system '/prompt.txt' \
    --deepspeed zero3_offload \
    --temperature 0.8 \
    --top_p 0.9 \
    --top_k 50 \
    --log_completions true \
    --async_generate false \
    --move_model_batches 16 \
    --offload_optimizer true \
    --offload_model true \
    --gc_collect_after_offload true \
    --sleep_level 1
