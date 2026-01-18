# 16GiB * 2
nproc_per_node=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model /DeepSeek-R1-7B \
    --model_type deepseek_r1_distill \
    --template deepseek_r1 \
    --train_type full \
    --dataset '/Data/CoT_data' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-5 \
    --report_to tensorboard \
    --gradient_accumulation_steps $(expr 128 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --max_length 3000 \
    --output_dir /LLM_DeepSeek_7B \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed /ds_config_zero2.json \