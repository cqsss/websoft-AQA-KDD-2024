Our solution uses a two-stage retrieval-reranking strategy, with the retrieval model being [NV-Embed-v1](https://huggingface.co/nvidia/NV-Embed-v1) and the reranking model being [glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat).


# Retrieval

1. Configure the Data Path:
     - Open the file `./retrieval/dataset_utils.py`.
     - Locate the variable `CORPUS_PATH` and `VALIDAT_PATH`. Update them with the paths to your dataset.

2. Configure the Output Path:
     - Open the files `./retrieval/nv_index_search.sh`.
     - Locate the variables `save_path` and `result_path` respectively. Update them with the desired output paths for the index and result files.

3. Run index and retrieval script:

```
bash ./retrieval/nv_index_search.sh
```

This script will generate the vector index and the top 100 retrieval results JSON file.

The generated JSON file contains the top 100 search results in the following format:

```
[["5390ac1820f70186a0eb4898", "5390b1d220f70186a0ee22f7", "64d641fe3fda6d7f06226e49", "5390a88c20f70186a0e99e25", "53e9ab5fb7602d97034f6761", ...],
...]
```

Our retrieval results are stored at `results\nv-embed-v1_title_abs_test.json`.

# Reranking