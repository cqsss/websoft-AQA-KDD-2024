run_glm() {
python run_rank_llm.py \
    --model_path=/home/model/glm-4-9b-chat \
    --top_k_candidates=100 \
    --retrieval_result_path=/path/to/nv-embed-v1_title_abs_test.json \
    --corpus_path=/path/to/data \
    --result_path=/path/to/output \
    --prompt_mode=rank_GPT  \
    --context_size=8192 \
    --variable_passages
}

run_glm