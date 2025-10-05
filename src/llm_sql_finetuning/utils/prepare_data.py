from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download
from datasets import Dataset
from transformers import AutoTokenizer


def _download_filename(filename: str, repo_id: str, workdir_path: str) -> Path:
    cached_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    local_path = Path(workdir_path, filename)
    shutil.copy(cached_path, local_path)
    return local_path


def prepare_load_dataset(
    train_file: str, val_file: str, test_file: str, repo_id: str, dataset_path: str
) -> tuple[Dataset, Dataset, Dataset]:
    path_list = train_file.split("/")[:-1]
    path = Path("/".join(path_list))
    path = Path(dataset_path) / Path(path)

    if path is not None:
        path.mkdir(parents=True, exist_ok=True)

    if not ((Path(dataset_path) / Path(train_file)).is_file()):
        train_file = str(_download_filename(train_file, repo_id, dataset_path))
    else:
        train_file = str(Path(dataset_path) / Path(train_file))

    if not ((Path(dataset_path) / Path(val_file)).is_file()):
        val_file = str(_download_filename(val_file, repo_id, dataset_path))
    else:
        val_file = str(Path(dataset_path) / Path(val_file))

    if not ((Path(dataset_path) / Path(test_file)).is_file()):
        test_file = str(_download_filename(test_file, repo_id, dataset_path))
    else:
        test_file = str(Path(dataset_path) / Path(test_file))

    return (
        Dataset.from_parquet(str(train_file)),
        Dataset.from_parquet(str(val_file)),
        Dataset.from_parquet(str(test_file)),
    )


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    template_name: str = "llmsql",
) -> Dataset:
    dataset = dataset.map(lambda x: _flatten_chat_messages(x, tokenizer, template_name=template_name))

    def tokenize_function(examples: dict[str, list[str]]) -> AutoTokenizer:
        return tokenizer(
            examples["text"],
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized_dataset


def _flatten_chat_messages(
    example: dict[str, list[list[dict[str, str]]]],
    tokenizer: AutoTokenizer,
    template_name: str = "llmsql",
) -> dict[str, str]:
    messages = example["messages"]
    template = tokenizer.chat_template[template_name]
    text = "".join([template[msg["role"]].format(content=msg["content"]) for msg in messages])
    return {"text": text}
