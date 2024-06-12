import json
import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import datasets
import faiss
import numpy as np
import torch
from nv_models import NVModel
from tqdm import tqdm
from transformers import HfArgumentParser

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Args:
    encoder: str = field(
        default="nvidia/NV-Embed-v1",
        metadata={'help': 'The encoder name or path.'}
    )
    fp16: bool = field(
        default=False,
        metadata={'help': 'Use fp16 in inference?'}
    )
    add_instruction: bool = field(
        default=True,
        metadata={'help': 'Add query-side instruction?'}
    )
    normalize_embeddings: bool = field(
        default=True,
        metadata={'help': 'Normalize embeddings?'}
    )
    pooling_method: str = field(
        default="cls",
        metadata={'help': 'Pooling method for encoding.'}
    )
    
    max_query_length: int = field(
        default=64,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=256,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=256,
        metadata={'help': 'Inference batch size.'}
    )
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    k: int = field(
        default=100,
        metadata={'help': 'How many neighbors to retrieve?'}
    )

    save_embedding: bool = field(
        default=False,
        metadata={'help': 'Save embeddings in memmap at save_dir?'}
    )
    load_embedding: bool = field(
        default=False,
        metadata={'help': 'Load embeddings from save_dir?'}
    )
    save_path: str = field(
        default="embeddings.memmap",
        metadata={'help': 'Path to save embeddings.'}
    )

    result_path: str = field(
        default="retrieval_results.json",
        metadata={'help': 'Path to save retrieval results.'}
    )



def index(model: NVModel, corpus: datasets.Dataset, batch_size: int = 256, max_length: int=4096, index_factory: str = "Flat", save_path: str = None, save_embedding: bool = False, load_embedding: bool = False):
    """
    1. Encode the entire corpus into dense embeddings; 
    2. Create faiss index; 
    3. Optionally save embeddings.
    """
    if load_embedding:
        test = model.encode("test")
        dtype = test.dtype
        dim = len(test)

        corpus_embeddings = np.memmap(
            save_path,
            mode="r",
            dtype=dtype
        ).reshape(-1, dim)
    
    else:
        # corpus_embeddings = model.encode_corpus(corpus["title"], batch_size=batch_size, max_length=max_length)
        corpus_embeddings = model.encode_corpus([f'title: {x["title"]}\nabstract: {x["abstract"]}' for x in corpus], batch_size=batch_size, max_length=max_length)
        dim = corpus_embeddings.shape[-1]
        
        if save_embedding:
            logger.info(f"saving embeddings at {save_path}...")
            memmap = np.memmap(
                save_path,
                shape=corpus_embeddings.shape,
                mode="w+",
                dtype=corpus_embeddings.dtype
            )

            length = corpus_embeddings.shape[0]
            # add in batch
            save_batch_size = 10000
            if length > save_batch_size:
                for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                    j = min(i + save_batch_size, length)
                    memmap[i: j] = corpus_embeddings[i: j]
            else:
                memmap[:] = corpus_embeddings
    
    # create faiss index
    faiss_index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)

    # if model.device == torch.device("cuda"):
        # co = faiss.GpuClonerOptions()
        # co = faiss.GpuMultipleClonerOptions()
        # co.useFloat16 = True
        # faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index, co)
        # faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    # NOTE: faiss only accepts float32
    logger.info("Adding embeddings...")
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    print(f"Index Dimension: {faiss_index.d}")
    return faiss_index


def search(model: NVModel, queries: datasets, faiss_index: faiss.Index, k:int = 100, batch_size: int = 256, max_length: int=4096):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    query_embeddings = model.encode_queries(queries["question"], batch_size=batch_size, max_length=max_length)
    query_size = len(query_embeddings)
    
    all_scores = []
    all_indices = []
    
    for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
        j = min(i + batch_size, query_size)
        query_embedding = query_embeddings[i: j]
        score, indice = faiss_index.search(query_embedding.astype(np.float32), k=k)
        all_scores.append(score)
        all_indices.append(indice)
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices
    

def makedirs(path):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return path

def save_json(obj, path:str):
    if not os.path.exists(path):
        makedirs(path)
    with open(path, "w") as f:
        return json.dump(obj, f, ensure_ascii=False)

def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]
    
    eval_data = datasets.load_dataset("dataset_untils.py", split="test", trust_remote_code=True)
    corpus = datasets.load_dataset("dataset_untils.py", split="train", trust_remote_code=True)
    
    model = NVModel(
        args.encoder, 
        query_instruction_for_retrieval=" Given a professional question, retrieve the most relevant papers to answer the questions: " if args.add_instruction else None
    )
    
    faiss_index = index(
        model=model, 
        corpus=corpus, 
        batch_size=args.batch_size,
        max_length=args.max_passage_length,
        index_factory=args.index_factory,
        save_path=args.save_path,
        save_embedding=args.save_embedding,
        load_embedding=args.load_embedding
    )
    
    scores, indices = search(
        model=model, 
        queries=eval_data, 
        faiss_index=faiss_index, 
        k=args.k, 
        batch_size=args.batch_size, 
        max_length=args.max_query_length
    )
    
    retrieval_results = []
    for indice in indices:
        # filter invalid indices
        indice = indice[indice != -1].tolist()
        retrieval_results.append(corpus[indice]["pid"])

    save_json(retrieval_results, args.result_path)


if __name__ == "__main__":
    main()