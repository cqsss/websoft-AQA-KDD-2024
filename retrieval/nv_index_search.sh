export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
python nv_index_search.py \
--encoder nvidia/NV-Embed-v1 \
--fp16 \
--save_embedding \
--save_path ./index/nv-embed-v1_title_abs_update_filter_embeddings.memmap \
--result_path ./results/nv-embed-v1_title_abs_test.json \
--batch_size 128 \
--k 100 \
--add_instruction \
--load_embedding \

