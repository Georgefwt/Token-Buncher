export N_GPUS=2
export BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
export DATA_DIR=./dataset/beavertails-qwen
export ROLLOUT_TP_SIZE=2
export PROJECTNAME=HarmfulRL_attack
export EXPERIMENT_NAME=harmfulrl-qwen-3b-grpoattack
export OUTPUT_DIR=./outputs/harmfulrl-qwen-3b-grpoattack
unset VLLM_ATTENTION_BACKEND # unnecessary for high version vllm

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p $OUTPUT_DIR
fi
bash ./configs/attack_obj1_grpo.sh


