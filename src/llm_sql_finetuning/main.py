from pathlib import Path

from llm_sql_finetuning.utils.prepare_data import (
    prepare_load_dataset,
    preprocess_dataset,
)
from llm_sql_finetuning.utils.run import parse_args_and_config

from trl import SFTTrainer
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
import torch


PATH = Path(__file__).resolve().parent.parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHAT_TEMPLATE = {
    "llmsql": {
        "user": "<|user|>{content}\n",
        "assistant": "<|assistant|>{content}\n",
    }
}


def main() -> None:
    args = parse_args_and_config()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.chat_template = CHAT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto" if DEVICE == "cuda" else None)
    training_args = TrainingArguments(
        output_dir=PATH / args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_dir=PATH / args.logging_dir,
        logging_steps=args.logging_steps,
        eval_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        fp16=True if DEVICE == "cuda" else False,
    )

    train, val, test = prepare_load_dataset(
        train_file=f"{args.shots}shot/train-00000-of-00001.parquet",
        val_file=f"{args.shots}shot/validation-00000-of-00001.parquet",
        test_file=f"{args.shots}shot/test-00000-of-00001.parquet",
        repo_id=args.repo_id,
        dataset_path=PATH / args.dataset_path,
    )

    train_ds = preprocess_dataset(train, tokenizer, args.max_seq_len)
    val_ds = preprocess_dataset(val, tokenizer, args.max_seq_len)

    trainer = SFTTrainer(
        model=model,
        eval_dataset=None if args.no_eval else val_ds,
        train_dataset=train_ds,
        args=training_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
