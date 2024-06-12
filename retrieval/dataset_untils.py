import json

import datasets
import jsonlines

CORPUS_PATH = "/AQA-test-public/pid_to_title_abs_update_filter.json"
TRAIN_PATH = "/qa_train.txt"
VALIDAT_PATH = "/AQA-test-public/qa_test_wo_ans_new.txt"

# 定义数据集中有哪些特征，及其类型
_FEATURES = datasets.Features(
    {
        "pid": datasets.Value("string"),
        "title": datasets.Value("string"),
        "abstract": datasets.Value("string"),
        "question": datasets.Value("string"),
        "pids": datasets.Sequence(datasets.Value("string")),
    },
)

class AQAKDD2024(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [datasets.BuilderConfig(name="default")]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description="None",
            features=_FEATURES,
            supervised_keys=None,
            homepage="None",
            license="None",
            citation="None",
        )

    def _split_generators(self, dl_manager):

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "split": "train",
                    "file_path": CORPUS_PATH,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "split": "validation",
                    "file_path": TRAIN_PATH,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "split": "test",
                    "file_path": VALIDAT_PATH,
                },
            ),
        ]

    def _generate_examples(self, split, file_path):
        if split == "train":
            with open(file_path, "r") as f:
                pid_to_title_abs = json.load(f)
                for pid, value in pid_to_title_abs.items():
                    title = value["title"] if value["title"] is not None else ""
                    abstract = value["abstract"] if value["abstract"] is not None else ""
                    title = title.replace("\n", " ").replace("\r", " ").replace("\t", " ")
                    abstract = abstract.replace("\n", " ").replace("\r", " ").replace("\t", " ")
                    yield pid, {
                        "pid": pid,
                        "title": title,
                        "abstract": abstract,
                    }

        elif split == "validation":
            with jsonlines.open(file_path, "r") as f:
                qid = 0
                for line in f:
                    qid += 1
                    yield qid, {
                        "question": line["question"],
                        "pids": line["pids"],
                    }

        elif split == "test":
            with jsonlines.open(file_path, "r") as f:
                qid = 0
                for line in f:
                    qid += 1
                    yield qid, {
                        "question": line["question"],
                    }