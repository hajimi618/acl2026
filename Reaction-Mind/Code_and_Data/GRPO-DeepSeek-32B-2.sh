# External vLLM

# Assume we have two nodes, one with 8 GPUs of 80GB each (880G) and another with 2 GPUs of 80GB each (2 80G).
#   NODE1. The node with 2*80G will be used to deploy the vLLM server.
#   NODE2. The node with 8*80G will be used for full-parameter fine-tuning of the 32B model.

# Note : Use beta=0 to disable the reference model; otherwise, it may lead to Out-of-Memory (OOM) errors.

# NODE1 for vLLM Server
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m swift.cli.rollout \
    --model /DeepSeek-R1-32B-CoT-SFT \
    --tensor_parallel_size 8