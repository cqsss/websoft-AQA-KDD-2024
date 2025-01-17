from typing import List

from rank_llm.rerank.rank_listwise_chat_llm import RankListwiseChatLLM
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.rerank.reranker import Reranker
from rank_llm.result import Result


class GLMReranker:
    def __init__(
        self,
        model_path: str = "/home/model/glm-4-9b-chat",
        context_size: int = 8192,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        device: str = "cuda",
        num_gpus: int = 1,
        variable_passages: bool = True,
        window_size: int = 20,
        system_message: str = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query",
    ) -> None:
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
        self._reranker = Reranker(agent)

    def rerank(
        self,
        retrieved_results: List[Result],
        rank_start: int = 0,
        rank_end: int = 100,
        window_size: int = 20,
        step: int = 10,
        shuffle_candidates: bool = False,
        logging: bool = False,
    ) -> List[Result]:
        """
        Reranks a list of retrieved results using the Zephyr model.

        Args:
            retrieved_results (List[Result]): The list of results to be reranked.
            rank_start (int, optional): The starting rank for processing. Defaults to 0.
            rank_end (int, optional): The end rank for processing. Defaults to 100.
            window_size (int, optional): The size of each sliding window. Defaults to 20.
            step (int, optional): The step size for moving the window. Defaults to 10.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before reranking. Defaults to False.
            logging (bool, optional): Enables logging of the reranking process. Defaults to False.

        Returns:
            List[Result]: A list containing the reranked results.

        Note:
            check 'rerank' for implementation details of reranking process.
        """
        return self._reranker.rerank(
            retrieved_results=retrieved_results,
            rank_start=rank_start,
            rank_end=rank_end,
            window_size=window_size,
            step=step,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
        )
