
from typing import List, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class NVModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            query_instruction_for_retrieval: str = None,
            use_fp16: bool = True
    ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, 
                                                trust_remote_code=True
                                                )
        self.query_instruction_for_retrieval = query_instruction_for_retrieval

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            use_fp16 = False
        if use_fp16: self.model.half()
        self.model = self.model.to(self.device)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            for module_key, module in self.model._modules.items():
                self.model._modules[module_key] = DataParallel(module)

    def encode_queries(self, queries: Union[List[str], str],
                       batch_size: int = 256,
                       max_length: int = 64,
                       convert_to_numpy: bool = True) -> np.ndarray:
        '''
        This function will be used for retrieval task
        '''
        query_prefix = ""
        self.model.eval()
        if self.query_instruction_for_retrieval is not None:
            query_prefix = "Instruct: " + self.query_instruction_for_retrieval + "\nQuery: "
        query_embeddings = self.model._do_encode(queries, instruction=query_prefix, batch_size=batch_size, max_length=max_length)
        query_embeddings = F.normalize(torch.tensor(query_embeddings), p=2, dim=1)
        query_embeddings = query_embeddings.cpu().numpy()
        return query_embeddings

    def encode_corpus(self,
                      corpus: Union[List[str], str],
                      batch_size: int = 256,
                      max_length: int = 4096,
                      convert_to_numpy: bool = True) -> np.ndarray:
        passage_prefix = ""
        self.model.eval()
        passage_embeddings = self.model._do_encode(corpus, instruction=passage_prefix, batch_size=batch_size, max_length=max_length)
        passage_embeddings = F.normalize(torch.tensor(passage_embeddings), p=2, dim=1)
        passage_embeddings = passage_embeddings.cpu().numpy()
        return passage_embeddings
    
    def encode(self,
            sentences: Union[List[str], str]) -> np.ndarray:
        prefix = ""
        max_length = 4096
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True
        embeddings = self.model.encode(sentences, instruction=prefix, max_length=max_length)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.cpu().numpy()
        if input_was_string:
            return embeddings[0]
        return embeddings