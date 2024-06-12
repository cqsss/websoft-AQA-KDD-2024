import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

import torch

from rank_llm.rerank.rankllm import PromptMode
from rank_llm.retrieve.retriever import RetrievalMode

from rank_llm.retrieve_and_rerank import retrieve_and_rerank


def main(args):
    model_path = args.model_path
    context_size = args.context_size
    top_k_candidates = args.top_k_candidates
    num_gpus = args.num_gpus
    dataset = args.retrieval_result_path
    corpus_path = args.corpus_path
    result_path = args.result_path
    prompt_mode = args.prompt_mode
    num_few_shot_examples = args.num_few_shot_examples
    shuffle_candidates = args.shuffle_candidates
    print_prompts_responses = args.print_prompts_responses
    num_few_shot_examples = args.num_few_shot_examples
    variable_passages = args.variable_passages
    retrieval_mode = RetrievalMode.AQA
    num_passes = args.num_passes
    step_size = args.step_size
    window_size = args.window_size
    system_message = args.system_message
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ = retrieve_and_rerank(
        model_path,
        dataset,
        corpus_path,
        result_path,
        retrieval_mode,
        top_k_candidates,
        context_size,
        device,
        num_gpus,
        prompt_mode,
        num_few_shot_examples,
        shuffle_candidates,
        print_prompts_responses,
        variable_passages=variable_passages,
        num_passes=num_passes,
        window_size=window_size,
        step_size=step_size,
        system_message=system_message,
    )


""" sample run:
python src/rank_llm/scripts/run_rank_llm.py  --model_path=castorini/rank_vicuna_7b_v1  --top_k_candidates=100 --dataset=dl20  --retrieval_method=SPLADE++_EnsembleDistil_ONNX --prompt_mode=rank_GPT  --context_size=4096 --variable_passages
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model. If `use_azure_ai`, pass your deployment name.",
    )
    parser.add_argument(
        "--context_size", type=int, default=4096, help="context size used for model"
    )
    parser.add_argument(
        "--top_k_candidates",
        type=int,
        default=100,
        help="the number of top candidates to rerank",
    )
    parser.add_argument(
        "--retrieval_result_path",
        type=str,
        required=True,
        help="the path of nv-embed-v1_title_abs_test.json"
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        required=True,
        help="the path of AQA corpus files"
    )
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="the save path of rerank result"
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="the number of GPUs to use"
    )
    parser.add_argument(
        "--prompt_mode",
        type=PromptMode,
        required=True,
        choices=list(PromptMode),
    )
    parser.add_argument(
        "--shuffle_candidates",
        action="store_true",
        help="whether to shuffle the candidates before reranking",
    )
    parser.add_argument(
        "--print_prompts_responses",
        action="store_true",
        help="whether to print promps and responses",
    )
    parser.add_argument(
        "--num_few_shot_examples",
        type=int,
        required=False,
        default=0,
        help="number of in context examples to provide",
    )
    parser.add_argument(
        "--variable_passages",
        action="store_true",
        help="whether the model can account for variable number of passages in input",
    )
    parser.add_argument(
        "--num_passes",
        type=int,
        required=False,
        default=1,
        help="number of passes to run the model",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="window size for the sliding window approach",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="step size for the sliding window approach",
    )
    parser.add_argument(
        "--system_message",
        type=str,
        default="You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
        help="the system message used in prompts",
    )
    args = parser.parse_args()
    main(args)
