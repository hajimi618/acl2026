nnodes=2
nproc_per_node=8


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
    --master_port 30600 \
    --nproc_per_node=$nproc_per_node \
    --nnodes=$nnodes \
    --node_rank=1 \
    --master_addr=xxx \
    /ms-swift-main/swift/cli/sft.py \
    --model /DeepSeek-R1-32B \
    --model_type deepseek_r1_distill \
    --template deepseek_r1 \
    --train_type full \
    --dataset '/Data/data_1' \
              '/Data/data_2' \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --report_to tensorboard \
    --gradient_accumulation_steps $(expr 64 / $nproc_per_node / $nnodes) \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --max_length 2300 \
    --output_dir /LLM_DeepSeek_32B \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed /ds_config_zero3.json \
    --loss_scale chem_tag \