import os
from typing import Any, Dict, List, Union

from rank_llm.rerank.rank_listwise_chat_llm import RankListwiseChatLLM
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.rerank.reranker import Reranker
from rank_llm.retrieve.retriever import RetrievalMode, Retriever


def retrieve_and_rerank(
    model_path: str,
    dataset: Union[str, List[str], List[Dict[str, Any]]],
    corpus_path: str,
    result_path: str,
    retrieval_mode: RetrievalMode,
    top_k_candidates: int = 100,
    context_size: int = 4096,
    device: str = "cuda",
    num_gpus: int = 1,
    prompt_mode: PromptMode = PromptMode.RANK_GPT,
    num_few_shot_examples: int = 0,
    shuffle_candidates: bool = False,
    print_prompts_responses: bool = False,
    variable_passages: bool = False,
    num_passes: int = 1,
    window_size: int = 20,
    step_size: int = 10,
    system_message: str = None,
):
    # Construct Rerank Agent
    if "glm" in model_path.lower():
        agent = RankListwiseChatLLM(
            model=model_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            num_few_shot_examples=num_few_shot_examples,
            device=device,
            num_gpus=num_gpus,
            variable_passages=variable_passages,
            window_size=window_size,
            system_message=system_message,
        )
    else:
        raise ValueError(f"Unsupported model: {model_path}")

    # Retrieve
    print("Retrieving:")
    if retrieval_mode == RetrievalMode.AQA:
        retrieved_results = Retriever.from_AQA_results(file_name=dataset, corpus=corpus_path)
    else:
        raise ValueError(f"Invalid retrieval mode: {retrieval_mode}")
    print("Reranking:")
    reranker = Reranker(agent)
    for pass_ct in range(num_passes):
        print(f"Pass {pass_ct + 1} of {num_passes}:")
        rerank_results = reranker.rerank(
            retrieved_results,
            rank_end=top_k_candidates,
            window_size=min(window_size, top_k_candidates),
            shuffle_candidates=shuffle_candidates,
            logging=print_prompts_responses,
            step=step_size,
        )
        if retrieval_mode == RetrievalMode.AQA:
            output = []
            for result in rerank_results:
                pids = [hit['docid'] for hit in result.hits]
                output.append(pids)
            with open(os.path.join(result_path, f"{model_path.split('/')[-1]}_rerank_nv-embed-v1_title_abs_test.txt"), 'w+') as f:
                for pids in output:
                    s = ','.join(pids[:20])
                    f.write(s + '\n')
        else:
            raise ValueError(f"Invalid retrieval mode: {retrieval_mode}")

    return rerank_results
