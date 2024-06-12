import json
import random
from typing import Dict, List, Optional, Tuple

import torch
from ftfy import fix_text
from transformers.generation import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from rank_llm.rerank.rankllm import PromptMode, RankLLM
from rank_llm.result import Result


class RankListwiseChatLLM(RankLLM):
    def __init__(
        self,
        model: str,
        context_size: int = 4096,
        prompt_mode: PromptMode = PromptMode.RANK_GPT,
        num_few_shot_examples: int = 0,
        device: str = "cuda",
        num_gpus: int = 1,
        variable_passages: bool = False,
        window_size: int = 20,
        system_message: str = None,
    ) -> None:
        """
         Creates instance of the RankListwiseOSLLM class, an extension of RankLLM designed for performing listwise ranking of passages using
         a specified language model. Advanced configurations are supported such as GPU acceleration, variable passage
         handling, and custom system messages for generating prompts.

         Parameters:
         - model (str): Identifier for the language model to be used for ranking tasks.
         - context_size (int, optional): Maximum number of tokens that can be handled in a single prompt. Defaults to 4096.
        - prompt_mode (PromptMode, optional): Specifies the mode of prompt generation, with the default set to RANK_GPT,
         indicating that this class is designed primarily for listwise ranking tasks following the RANK_GPT methodology.
         - num_few_shot_examples (int, optional): Number of few-shot learning examples to include in the prompt, allowing for
         the integration of example-based learning to improve model performance. Defaults to 0, indicating no few-shot examples
         by default.
         - device (str, optional): Specifies the device for model computation ('cuda' for GPU or 'cpu'). Defaults to 'cuda'.
         - num_gpus (int, optional): Number of GPUs to use for model loading and inference. Defaults to 1.
         - variable_passages (bool, optional): Indicates whether the number of passages to rank can vary. Defaults to False.
         - window_size (int, optional): The window size for handling text inputs. Defaults to 20.
         - system_message (Optional[str], optional): Custom system message to be included in the prompt for additional
         instructions or context. Defaults to None.

         Raises:
         - AssertionError: If CUDA is specified as the device but is not available on the system.
         - ValueError: If an unsupported prompt mode is provided.

         Note:
         - This class is operates given scenarios where listwise ranking is required, with support for dynamic
         passage handling and customization of prompts through system messages and few-shot examples.
         - GPU acceleration is supported and recommended for faster computations.
        """
        super().__init__(model, context_size, prompt_mode, num_few_shot_examples)
        self._device = device
        if self._device == "cuda":
            assert torch.cuda.is_available()
        if prompt_mode != PromptMode.RANK_GPT:
            raise ValueError(
                f"Unsupported prompt mode: {prompt_mode}. The only prompt mode currently supported is a slight variation of Rank_GPT prompt."
            )
        # ToDo: Make repetition_penalty configurable
        self._llm = AutoModelForCausalLM.from_pretrained(model,
                                                         torch_dtype=torch.bfloat16,
                                                         low_cpu_mem_usage=True,
                                                         trust_remote_code=True
                                                         ).to(device).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        self._variable_passages = variable_passages
        self._window_size = window_size
        self._system_message = system_message
        self._output_token_estimate = None
        if num_few_shot_examples > 0:
            with open("data/output_v2_aug_filtered.jsonl", "r") as json_file:
                self._examples = list(json_file)[1:-1]

    def run_llm(
        self, prompt: List[Dict[str, str]], current_window_size: Optional[int] = None
    ) -> Tuple[str, int]:
        if current_window_size is None:
            current_window_size = self._window_size
        inputs = self._tokenizer.apply_chat_template(prompt,
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       ).to(self._device)
        gen_cfg = GenerationConfig.from_model_config(self._llm.config)
        gen_cfg.max_new_tokens = self.num_output_tokens(current_window_size)
        gen_cfg.min_new_tokens = self.num_output_tokens(current_window_size)
        # gen_cfg.temperature = 0
        gen_cfg.do_sample = False
        output_ids = self._llm.generate(**inputs, generation_config=gen_cfg)

        if self._llm.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = self._tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        # print(outputs)
        return outputs, output_ids.size(0)

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size
        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate
        else:
            _output_token_estimate = (
                len(
                    self._tokenizer.encode(
                        " > ".join([f"[{i+1}]" for i in range(current_window_size)])
                    )
                )
                - 1
            )
            if (
                self._output_token_estimate is None
                and self._window_size == current_window_size
            ):
                self._output_token_estimate = _output_token_estimate
            return _output_token_estimate

    def _add_prefix_prompt(self, query: str, num: int) -> str:
        return f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"

    def _add_post_prompt(self, query: str, num: int) -> str:
        example_ordering = "[2] > [1]" if self._variable_passages else "[4] > [2]"
        return f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}, Only respond with the ranking results, do not say any word or explain."

    def _add_few_shot_examples(self, conv):
        for _ in range(self._num_few_shot_examples):
            ex = random.choice(self._examples)
            obj = json.loads(ex)
            prompt = obj["conversations"][0]["value"]
            response = obj["conversations"][1]["value"]
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], response)
        return conv

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[List[Dict[str, str]], int]:
        query = fix_text(result.query)
        num = len(result.hits[rank_start:rank_end])
        # max_length = 300 * (20 / (rank_end - rank_start))
        max_length = 1000
        while True:
            messages = [{'role': 'user',
                        'content': "You are RankGLM, an intelligent assistant that can rank passages based on their relevancy to the query."},
                        {'role': 'user',
                         'content': self._add_prefix_prompt(query, num)},
                        {'role': 'assistant', 
                         'content': 'Okay, please provide the passages.'}
                        ]
            rank = 0
            for hit in result.hits[rank_start:rank_end]:
                rank += 1
                content = hit["content"]
                content = content.replace("Title: Content: ", "")
                content = content.strip()
                # For Japanese should cut by character: content = content[:int(max_length)]
                content = " ".join(content.split()[: int(max_length)])
                messages.append({'role': 'user', 'content': f"[{rank}] {fix_text(self._replace_number(content))}"})
                messages.append({'role': 'assistant', 'content': f"Received passage [{rank}]"})
            messages.append({'role': 'user', 'content': self._add_post_prompt(query, num)})
            # conv.append_message(conv.roles[0], input_context)
            # conv.append_message(conv.roles[1], None)
            # prompt = conv.get_prompt()
            # prompt = fix_text(prompt)
            num_tokens = self.get_num_tokens(messages)
            if num_tokens <= self.max_tokens() - self.num_output_tokens(
                rank_end - rank_start
            ):
                break
            else:
                max_length -= max(
                    1,
                    (
                        num_tokens
                        - self.max_tokens()
                        + self.num_output_tokens(rank_end - rank_start)
                    )
                    // ((rank_end - rank_start) * 4),
                )
        return messages, self.get_num_tokens(messages)

    def get_num_tokens(self, prompt: List[Dict[str, str]]) -> int:
        return len(self._tokenizer.encode(self._tokenizer.apply_chat_template(prompt, add_generation_prompt=True,tokenize=False)))

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0
