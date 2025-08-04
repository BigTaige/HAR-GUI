export NCCL_DEBUG=WARN
export WORLD_SIZE=1
#export MASTER_PORT="12339"
CURRENT_IP=$(hostname -I | awk '{print $1}')
MASTER_ADDR=$CURRENT_IP \
# NPROC_PER_NODE=16 \
CUDA_VISIBLE_DEVICES=1 \
SIZE_FACTOR=8 MAX_PIXELS=1605632 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#--model_type qwen2vl \

swift infer \
    --model_type qwen2_5_vl \
    --model ./models/Qwen2.5-VL-72B-Instruct or ./models/HAR-GUI-3B \
    --temperature 0.0 \
    --max_new_tokens 2048 \
    --val_dataset your_data_path.json \
    --infer_backend vllm \
    --gpu_memory_utilization 0.7 \
    --max_model_len 30000 \
    --result_path output_dir/output.jsonl \
    --tensor_parallel_size 8 \

# nohup sh swift_inference.sh > logs/inferenct_log.txt &

# For data format, please refer to swift doc.




