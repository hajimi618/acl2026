export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nproc_per_node=8

#export NCCL_TIMEOUT=7200

#export NCCL_BLOCKING_WAIT=1
#export NCCL_ASYNC_ERROR_HANDLING=1

#export TORCH_NCCL_TIMEOUT=7200
#export TORCH_NCCL_BLOCKING_WAIT=1
#export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
swift sft \
    --model /Qwen3-32B \
    --train_type full\
    --dataset '/data/data-RAG' \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 64 / $nproc_per_node) \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir /Qwen3_32B_full \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.15 \
    --dataloader_num_workers 8 \
    --model_author swift \
    --model_name swift-robot \
    --deepspeed zero3

