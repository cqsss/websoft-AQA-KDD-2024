import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

from rank_llm.result import Result


class RetrievalMode(Enum):
    AQA = "aqa"

    def __str__(self):
        return self.value


class Retriever:
    def __init__(
        self,
        retrieval_mode: RetrievalMode,
        dataset: Union[str, List[str], List[Dict[str, Any]]],
    ) -> None:
        self._retrieval_mode = retrieval_mode
        self._dataset = dataset

    @staticmethod
    def from_AQA_results(file_name: str, corpus: str):
        """
        Creates a Retriever instance from saved retrieval results specified by 'file_name'.

        Args:
            file_name (str): The file name containing the saved retrieval results.

        Returns:
            List[Dict[str, Any]]: The retrieval results loaded from the file.

        Raises:
            ValueError: If the file content is not in the expected format.
        """
        with open(file_name, "r") as f:
            retrieved_results = json.load(f)
        if not isinstance(retrieved_results, list):
            raise ValueError(
                f"Invalid retrieval format: Expected a list of dictionaries, got {type(retrieved_results)}"
            )
        retriever = Retriever(RetrievalMode.AQA, dataset=retrieved_results)
        return retriever.retrieve(corpus)


    def retrieve(
        self, corpus: str
    ) -> List[Dict[str, Any]]:
        if self._retrieval_mode == RetrievalMode.AQA:
            # [[top 100 candidate id for query 1], [...], ...]
            # test
            with open(os.path.join(corpus, 'pid_to_title_abs_update_filter.json'), 'r') as f:
                p2content = json.load(f)
            with open(os.path.join(corpus, 'qa_test_wo_ans_new.txt'), 'r') as f:
                query_list = []
                for line in f.readlines():
                    query_list.append(json.loads(line.strip()))
            retrieved_results = [
                Result(query=query_list[idx]['question'], hits= [
                    {
                        "content": f'Title: {p2content[pid]["title"]}\nAbstract: {p2content[pid]["abstract"]}',
                        "qid": idx,
                        "docid": pid,
                        "rank": i + 1,
                        "score": i + 1,
                    } for i, pid in enumerate(pid_list)
                ]) for idx, pid_list in enumerate(self._dataset)
            ]
        else:
            raise ValueError(f"Invalid retrieval mode: {self._retrieval_mode}")
        for result in retrieved_results:
            self._validate_result(result)
        return retrieved_results

    def _validate_result(self, result: Result):
        if not isinstance(result, Result):
            raise ValueError(
                f"Invalid result format: Expected type `Result`, got {type(result)}"
            )
        if not result.query:
            raise ValueError(f"Invalid format: missing `query`")
        if not result.hits:
            raise ValueError(f"Invalid format: missing `hits`")
        for hit in result.hits:
            if not isinstance(hit, Dict):
                raise ValueError(
                    f"Invalid hits format: Expected a list of Dicts where each Dict represents a hit."
                )
            if "content" not in hit.keys():
                raise ValueError((f"Invalid format: Missing `content` key in hit"))
