diff --git a/README.md b/README.md
index bf2981d..4e23ce6 100644
--- a/README.md
+++ b/README.md
@@ -440,7 +440,7 @@ mlx_lm_lora.train \
 --model <model_path>              # Model path or HF repo
 --data <data_path>                # Dataset path or HF dataset name
 --train-type lora                 # lora, dora, or full
---train-mode sft                  # sft, dpo, cpo, orpo, grpo, etc.
+--train-mode sft                  # sft, dpo, cpo, orpo, grpo, distill_on_policy, etc.
 
 # Training schedule
 --batch-size 4                    # Batch size
@@ -464,6 +464,13 @@ mlx_lm_lora.train \
 --sgd-nesterov                   # Enable Nesterov momentum for SGD
 --grad-checkpoint                # Enable gradient checkpointing
 
+# Distillation
+--teacher-model <path>           # Teacher checkpoint for distillation
+--distill-prompts-data <path>    # Prompt-only dataset for distillation rollouts
+--training-schedule sft:0.7,distill_on_policy:0.3  # Mix modes by fixed proportions
+--max-generation-tokens 256      # Truncate student rollouts after this many tokens
+--distill-temperature 0.0        # Sampling temperature for student rollouts
+
 # Quantization
 --load-in-4bits                  # 4-bit quantization
 --load-in-6bits                  # 6-bit quantization  
@@ -777,6 +784,36 @@ mlx_lm_lora.train \
 --weight-decay 0.01               # Applies to optimizers with built-in weight decay support
 ```
 
+### Training Schedules
+
+```shell
+--training-schedule sft:0.8,distill_on_policy:0.2
+```
+
+- Allocates the global `--iters` budget proportionally across multiple training modes.
+- Each block runs sequentially in the order provided by the schedule list.
+- Combine with any supported training mode (currently SFT + on-policy distillation).
+
+### On-Policy Distillation
+
+```shell
+mlx_lm_lora.train \
+  --train-mode distill_on_policy \
+  --teacher-model <teacher_path> \
+  --distill-prompts-data <prompts_dir> \
+  --max-generation-tokens 256 \
+  --distill-temperature 0.0
+```
+
+- Requires the teacher to share an identical tokenizer/vocabulary with the student.
+- The teacher logits are computed on the student's rollouts; tokenizer vocabularies and token→id mappings must match (including EOS/BOS/PAD/UNK ids).
+- The prompts dataset should be prompt-only JSONL (chat-style `messages` or `prompt`) with `train.jsonl` at minimum.
+- Student rollouts are truncated at `--max-generation-tokens` without forcing EOS.
+- Distillation automatically uses `--gradient-accumulation-steps 1` during the distill phase regardless of outer settings.
+- Leave headroom in `--max-seq-length` for generated tokens; the trainer clips generation to stay within the model context window.
+- During distillation the student switches to evaluation mode for generation and returns to training immediately afterwards.
+- Provide a `--training-schedule` entry (e.g., `sft:0.7,distill_on_policy:0.3`) to interleave distillation with other modes; omit it for pure distillation.
+
 ### Reward Function System (GRPO)
 
 List available reward functions:
diff --git a/feature-on-policy-distillation-diff.md b/feature-on-policy-distillation-diff.md
new file mode 100644
index 0000000..ba6ea8c
--- /dev/null
+++ b/feature-on-policy-distillation-diff.md
@@ -0,0 +1,1847 @@
+diff --git a/README.md b/README.md
+index bf2981d..171eb9e 100644
+--- a/README.md
++++ b/README.md
+@@ -440,7 +440,7 @@ mlx_lm_lora.train \
+ --model <model_path>              # Model path or HF repo
+ --data <data_path>                # Dataset path or HF dataset name
+ --train-type lora                 # lora, dora, or full
+---train-mode sft                  # sft, dpo, cpo, orpo, grpo, etc.
++--train-mode sft                  # sft, dpo, cpo, orpo, grpo, distill_on_policy, etc.
+ 
+ # Training schedule
+ --batch-size 4                    # Batch size
+@@ -464,6 +464,13 @@ mlx_lm_lora.train \
+ --sgd-nesterov                   # Enable Nesterov momentum for SGD
+ --grad-checkpoint                # Enable gradient checkpointing
+ 
++# Distillation
++--teacher-model <path>           # Teacher checkpoint for distillation
++--distill-prompts-data <path>    # Prompt-only dataset for distillation rollouts
++--training-schedule sft:0.7,distill_on_policy:0.3  # Mix modes by fixed proportions
++--max-generation-tokens 256      # Truncate student rollouts after this many tokens
++--distill-temperature 0.0        # Sampling temperature for student rollouts
++
+ # Quantization
+ --load-in-4bits                  # 4-bit quantization
+ --load-in-6bits                  # 6-bit quantization  
+@@ -777,6 +784,35 @@ mlx_lm_lora.train \
+ --weight-decay 0.01               # Applies to optimizers with built-in weight decay support
+ ```
+ 
++### Training Schedules
++
++```shell
++--training-schedule sft:0.8,distill_on_policy:0.2
++```
++
++- Allocates the global `--iters` budget proportionally across multiple training modes.
++- Each block runs sequentially in the order provided by the schedule list.
++- Combine with any supported training mode (currently SFT + on-policy distillation).
++
++### On-Policy Distillation
++
++```shell
++mlx_lm_lora.train \
++  --train-mode distill_on_policy \
++  --teacher-model <teacher_path> \
++  --distill-prompts-data <prompts_dir> \
++  --max-generation-tokens 256 \
++  --distill-temperature 0.0
++```
++
++- Requires the teacher to share an identical tokenizer/vocabulary with the student.
++- The teacher logits are computed on the student's rollouts; tokenizer vocabularies and token→id mappings must match (including EOS/BOS/PAD/UNK ids).
++- The prompts dataset should be prompt-only JSONL (chat-style `messages` or `prompt`) with `train.jsonl` at minimum.
++- Student rollouts are truncated at `--max-generation-tokens` without forcing EOS.
++- Distillation automatically uses `--gradient-accumulation-steps 1` during the distill phase regardless of outer settings.
++- Leave headroom in `--max-seq-length` for generated tokens; the trainer clips generation to stay within the model context window.
++- Provide a `--training-schedule` entry (e.g., `sft:0.7,distill_on_policy:0.3`) to interleave distillation with other modes; omit it for pure distillation.
++
+ ### Reward Function System (GRPO)
+ 
+ List available reward functions:
+diff --git a/feature-on-policy-distillation-diff.md b/feature-on-policy-distillation-diff.md
+new file mode 100644
+index 0000000..338b8ba
+--- /dev/null
++++ b/feature-on-policy-distillation-diff.md
+@@ -0,0 +1,888 @@
++diff --git a/README.md b/README.md
++index bf2981d..d4bb39c 100644
++--- a/README.md
+++++ b/README.md
++@@ -440,7 +440,7 @@ mlx_lm_lora.train \
++ --model <model_path>              # Model path or HF repo
++ --data <data_path>                # Dataset path or HF dataset name
++ --train-type lora                 # lora, dora, or full
++---train-mode sft                  # sft, dpo, cpo, orpo, grpo, etc.
+++--train-mode sft                  # sft, dpo, cpo, orpo, grpo, distill_on_policy, etc.
++ 
++ # Training schedule
++ --batch-size 4                    # Batch size
++@@ -464,6 +464,13 @@ mlx_lm_lora.train \
++ --sgd-nesterov                   # Enable Nesterov momentum for SGD
++ --grad-checkpoint                # Enable gradient checkpointing
++ 
+++# Distillation
+++--teacher-model <path>           # Teacher checkpoint for distillation
+++--distill-prompts-data <path>    # Prompt-only dataset for distillation rollouts
+++--training-schedule sft:0.7,distill_on_policy:0.3  # Mix modes by fixed proportions
+++--max-generation-tokens 256      # Truncate student rollouts after this many tokens
+++--distill-temperature 0.0        # Sampling temperature for student rollouts
+++
++ # Quantization
++ --load-in-4bits                  # 4-bit quantization
++ --load-in-6bits                  # 6-bit quantization  
++@@ -777,6 +784,34 @@ mlx_lm_lora.train \
++ --weight-decay 0.01               # Applies to optimizers with built-in weight decay support
++ ```
++ 
+++### Training Schedules
+++
+++```shell
+++--training-schedule sft:0.8,distill_on_policy:0.2
+++```
+++
+++- Allocates the global `--iters` budget proportionally across multiple training modes.
+++- Each block runs sequentially in the order provided by the schedule list.
+++- Combine with any supported training mode (currently SFT + on-policy distillation).
+++
+++### On-Policy Distillation
+++
+++```shell
+++mlx_lm_lora.train \
+++  --train-mode distill_on_policy \
+++  --teacher-model <teacher_path> \
+++  --distill-prompts-data <prompts_dir> \
+++  --max-generation-tokens 256 \
+++  --distill-temperature 0.0
+++```
+++
+++- Requires the teacher to share an identical tokenizer/vocabulary with the student.
+++- The teacher logits are computed on the student's rollouts; ensure the tokenizer vocabularies align.
+++- The prompts dataset should be prompt-only JSONL (chat-style `messages` or `prompt`) with `train.jsonl` at minimum.
+++- Student rollouts are truncated at `--max-generation-tokens` without forcing EOS.
+++- Distillation automatically uses `--gradient-accumulation-steps 1` during the distill phase regardless of outer settings.
+++- Provide a `--training-schedule` entry (e.g., `sft:0.7,distill_on_policy:0.3`) to interleave distillation with other modes; omit it for pure distillation.
+++
++ ### Reward Function System (GRPO)
++ 
++ List available reward functions:
++diff --git a/mlx_lm_lora/train.py b/mlx_lm_lora/train.py
++index 9efa33a..7a4662b 100644
++--- a/mlx_lm_lora/train.py
+++++ b/mlx_lm_lora/train.py
++@@ -2,9 +2,9 @@ from pathlib import Path
++ import importlib.util
++ import argparse
++ import math
++-import yaml
++ import sys
++ import re
+++import yaml
++ 
++ import numpy as np
++ 
++@@ -26,7 +26,8 @@ from .trainer.rflhf_trainer import RLHFTrainingArgs, evaluate_rlhf, train_rlhf
++ from .trainer.xpo_trainer import  XPOTrainingArgs, evaluate_xpo, train_xpo
++ from .trainer.dpo_trainer import DPOTrainingArgs, evaluate_dpo, train_dpo
++ from .trainer.cpo_trainer import CPOTrainingArgs, evaluate_cpo, train_cpo
++-from .trainer.datasets import CacheDataset, load_dataset
+++from .trainer.distill_trainer import DistillationTrainingArgs, train_on_policy_distill
+++from .trainer.datasets import CacheDataset, load_dataset, load_prompt_only_dataset
++ from .utils import fuse_and_save_model, from_pretrained
++ 
++ from mlx_lm.tuner.utils import (
++@@ -70,6 +71,7 @@ CONFIG_DEFAULTS = {
++         "adafactor": {},
++     },
++     "data": "data/",
+++    "distill_prompts_data": None,
++     "seed": 0,
++     "num_layers": 16,
++     "batch_size": 4,
++@@ -86,9 +88,11 @@ CONFIG_DEFAULTS = {
++     "test": False,
++     "test_batches": 500,
++     "max_seq_length": 2048,
+++    "max_generation_tokens": 256,
++     "config": None,
++     "grad_checkpoint": False,
++     "lr_schedule": None,
+++    "training_schedule": None,
++     "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 10.0},
++     "mask_prompt": False,
++     "fuse": True,
++@@ -106,6 +110,8 @@ CONFIG_DEFAULTS = {
++     "judge": None,
++     "judge_config": {},
++     "alpha": 1e-5,
+++    "teacher_model": None,
+++    "distill_temperature": 0.0,
++ 
++     # GRPO args
++     "group_size": 4,
++@@ -147,6 +153,65 @@ def calculate_iters(train_set, batch_size, epochs) -> int:
++     return iters
++ 
++ 
+++def parse_training_schedule(schedule_str: str):
+++    if schedule_str is None:
+++        return None
+++    entries = []
+++    for raw_part in schedule_str.split(","):
+++        part = raw_part.strip()
+++        if not part:
+++            continue
+++        if ":" not in part:
+++            raise ValueError(
+++                f"Invalid training schedule entry '{part}'. Expected format mode:weight."
+++            )
+++        mode, weight = part.split(":", 1)
+++        try:
+++            weight_val = float(weight.strip())
+++        except ValueError as exc:
+++            raise ValueError(f"Invalid weight '{weight}' in training schedule.") from exc
+++        entries.append({"mode": mode.strip(), "weight": weight_val})
+++    if not entries:
+++        raise ValueError("Training schedule was provided but no valid entries were parsed.")
+++    if any(item["weight"] <= 0 for item in entries):
+++        raise ValueError("Training schedule weights must be positive.")
+++    return entries
+++
+++
+++def allocate_schedule_iterations(total_iters: int, entries):
+++    if total_iters is None:
+++        raise ValueError("Total iterations must be specified when using a training schedule.")
+++    total_weight = sum(item["weight"] for item in entries)
+++    if total_weight <= 0:
+++        raise ValueError("Training schedule weights must sum to a positive value.")
+++
+++    exact_counts = [
+++        (item, total_iters * item["weight"] / total_weight) for item in entries
+++    ]
+++    assigned_counts = []
+++    residuals = []
+++    for item, exact in exact_counts:
+++        count = int(math.floor(exact))
+++        assigned_counts.append([item, count])
+++        residuals.append((exact - count, item))
+++
+++    assigned_total = sum(count for _, count in assigned_counts)
+++    remaining = total_iters - assigned_total
+++    residuals.sort(key=lambda tup: tup[0], reverse=True)
+++
+++    idx = 0
+++    while remaining > 0 and residuals:
+++        _, item = residuals[idx % len(residuals)]
+++        for entry in assigned_counts:
+++            if entry[0] is item:
+++                entry[1] += 1
+++                remaining -= 1
+++                break
+++        idx += 1
+++
+++    return [(entry["mode"], count) for entry, count in assigned_counts]
+++
+++
++ def build_parser():
++     parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
++     parser.add_argument(
++@@ -154,6 +219,12 @@ def build_parser():
++         type=str,
++         help="The path to the local model directory or Hugging Face repo.",
++     )
+++    parser.add_argument(
+++        "--teacher-model",
+++        type=str,
+++        default=None,
+++        help="Optional teacher model path for on-policy distillation.",
+++    )
++     parser.add_argument(
++         "--load-in-4bits",
++         action="store_true",
++@@ -188,6 +259,12 @@ def build_parser():
++             "of a Hugging Face dataset (e.g., 'mlx-community/wikisql')"
++         ),
++     )
+++    parser.add_argument(
+++        "--distill-prompts-data",
+++        type=str,
+++        default=None,
+++        help="Directory or dataset id providing prompts for on-policy distillation.",
+++    )
++     parser.add_argument(
++         "--train-type",
++         type=str,
++@@ -198,8 +275,14 @@ def build_parser():
++         "--train-mode",
++         type=str,
++         default="sft",
++-        choices=["sft", "dpo", "cpo", "orpo", "grpo", "online_dpo", "xpo", "rlhf"],
++-        help="Training mode: sft, dpo, rlhf, online_dpo, xpo, cpo, orpo, or grpo, default is sft",
+++        choices=["sft", "dpo", "cpo", "orpo", "grpo", "online_dpo", "xpo", "rlhf", "distill_on_policy"],
+++        help="Training mode: sft, dpo, rlhf, online_dpo, xpo, cpo, orpo, grpo, or distill_on_policy. Default is sft",
+++    )
+++    parser.add_argument(
+++        "--training-schedule",
+++        type=str,
+++        default=None,
+++        help="Comma-separated list of mode:weight pairs to sequence training phases.",
++     )
++     parser.add_argument(
++         "--optimizer",
++@@ -295,6 +378,18 @@ def build_parser():
++         type=int,
++         help="Maximum sequence length.",
++     )
+++    parser.add_argument(
+++        "--max-generation-tokens",
+++        type=int,
+++        default=256,
+++        help="Maximum number of tokens to generate during on-policy distillation.",
+++    )
+++    parser.add_argument(
+++        "--distill-temperature",
+++        type=float,
+++        default=0.0,
+++        help="Sampling temperature when generating student rollouts for distillation.",
+++    )
++     parser.add_argument(
++         "-c",
++         "--config",
++@@ -455,6 +550,8 @@ def train_model(
++     train_set,
++     valid_set,
++     training_callback: TrainingCallback = None,
+++    teacher_model: nn.Module = None,
+++    distill_dataset=None,
++ ):
++     mx.random.seed(args.seed)
++ 
++@@ -567,6 +664,71 @@ def train_model(
++ 
++     opt = opt_class(learning_rate=lr, **optimizer_config)
++ 
+++    schedule_entries = getattr(args, "_parsed_training_schedule", None)
+++    if schedule_entries:
+++        iteration_plan = allocate_schedule_iterations(args.iters, schedule_entries)
+++        sft_train_cache = CacheDataset(train_set) if train_set and len(train_set) else None
+++        sft_valid_cache = CacheDataset(valid_set) if valid_set and len(valid_set) else None
+++        distill_cache = CacheDataset(distill_dataset) if distill_dataset and len(distill_dataset) else None
+++
+++        for mode, count in iteration_plan:
+++            if count <= 0:
+++                continue
+++            if mode == "sft":
+++                if sft_train_cache is None or sft_valid_cache is None:
+++                    raise ValueError("SFT dataset is required but not available for the training schedule.")
+++                sft_training_args = SFTTrainingArgs(
+++                    batch_size=args.batch_size,
+++                    iters=count,
+++                    val_batches=args.val_batches,
+++                    steps_per_report=args.steps_per_report,
+++                    steps_per_eval=args.steps_per_eval,
+++                    steps_per_save=args.save_every,
+++                    adapter_file=adapter_file,
+++                    max_seq_length=args.max_seq_length,
+++                    grad_checkpoint=args.grad_checkpoint,
+++                    gradient_accumulation_steps=args.gradient_accumulation_steps,
+++                )
+++                train_sft(
+++                    model=model,
+++                    args=sft_training_args,
+++                    optimizer=opt,
+++                    train_dataset=sft_train_cache,
+++                    val_dataset=sft_valid_cache,
+++                    training_callback=training_callback,
+++                )
+++            elif mode == "distill_on_policy":
+++                if teacher_model is None:
+++                    raise ValueError("Teacher model required for distillation in training schedule.")
+++                if distill_cache is None:
+++                    raise ValueError("Distillation dataset required for training schedule entry.")
+++                distill_args = DistillationTrainingArgs(
+++                    batch_size=args.batch_size,
+++                    iters=count,
+++                    val_batches=0,
+++                    steps_per_report=args.steps_per_report,
+++                    steps_per_eval=args.steps_per_eval,
+++                    steps_per_save=args.save_every,
+++                    adapter_file=adapter_file,
+++                    max_seq_length=args.max_seq_length,
+++                    grad_checkpoint=False,
+++                    gradient_accumulation_steps=args.gradient_accumulation_steps,
+++                    max_generation_tokens=args.max_generation_tokens,
+++                    distill_temperature=args.distill_temperature,
+++                )
+++                train_on_policy_distill(
+++                    model=model,
+++                    teacher_model=teacher_model,
+++                    tokenizer=tokenizer,
+++                    optimizer=opt,
+++                    train_dataset=distill_cache,
+++                    args=distill_args,
+++                    training_callback=training_callback,
+++                )
+++            else:
+++                raise ValueError(f"Unsupported mode '{mode}' in training schedule.")
+++        return
+++
++     if args.train_mode == "orpo":
++         orpo_training_args = ORPOTrainingArgs(
++             batch_size=args.batch_size,
++@@ -864,6 +1026,35 @@ def train_model(
++             training_callback=training_callback,
++         )
++ 
+++    elif args.train_mode == "distill_on_policy":
+++        if teacher_model is None:
+++            raise ValueError("Teacher model must be provided for distill_on_policy training.")
+++        if distill_dataset is None or len(distill_dataset) == 0:
+++            raise ValueError("A non-empty distillation dataset is required for distill_on_policy training.")
+++        distill_args = DistillationTrainingArgs(
+++            batch_size=args.batch_size,
+++            iters=args.iters,
+++            val_batches=0,
+++            steps_per_report=args.steps_per_report,
+++            steps_per_eval=args.steps_per_eval,
+++            steps_per_save=args.save_every,
+++            adapter_file=adapter_file,
+++            max_seq_length=args.max_seq_length,
+++            grad_checkpoint=False,
+++            gradient_accumulation_steps=args.gradient_accumulation_steps,
+++            max_generation_tokens=args.max_generation_tokens,
+++            distill_temperature=args.distill_temperature,
+++        )
+++        train_on_policy_distill(
+++            model=model,
+++            teacher_model=teacher_model,
+++            tokenizer=tokenizer,
+++            optimizer=opt,
+++            train_dataset=CacheDataset(distill_dataset),
+++            args=distill_args,
+++            training_callback=training_callback,
+++        )
+++
++     elif args.train_mode == "sft":
++         sft_training_args = SFTTrainingArgs(
++             batch_size=args.batch_size,
++@@ -935,6 +1126,9 @@ def evaluate_model(args, model: nn.Module, tokenizer, test_set):
++         for metric_name, metric_value in test_metrics.items():
++             print(f"  {metric_name}: {float(metric_value):.3f}")
++ 
+++    elif args.train_mode == "distill_on_policy":
+++        print("Evaluation for distill_on_policy training is not currently implemented.")
+++
++     elif args.train_mode == "rlhf":
++         if args.reference_model_path:
++             reference_model, _ = load(args.reference_model_path)
++@@ -1085,6 +1279,52 @@ def run(args, training_callback: TrainingCallback = None):
++         quantized_load=quanziation_config,
++     )
++ 
+++    schedule_entries = None
+++    raw_schedule = getattr(args, "training_schedule", None)
+++    if raw_schedule:
+++        if isinstance(raw_schedule, str):
+++            schedule_entries = parse_training_schedule(raw_schedule)
+++        elif isinstance(raw_schedule, list):
+++            schedule_entries = raw_schedule
+++        else:
+++            raise ValueError("training_schedule must be a string or list of schedule entries.")
+++    setattr(args, "_parsed_training_schedule", schedule_entries)
+++
+++    requires_distill = args.train_mode == "distill_on_policy"
+++    if schedule_entries:
+++        requires_distill = requires_distill or any(
+++            entry["mode"] == "distill_on_policy" for entry in schedule_entries
+++        )
+++
+++    teacher_model = None
+++    distill_dataset = None
+++    if args.train and requires_distill:
+++        if args.teacher_model is None:
+++            raise ValueError("A teacher model must be provided when using on-policy distillation.")
+++        teacher_model, teacher_tokenizer = from_pretrained(args.teacher_model)
+++
+++        def _extract_vocab(tok):
+++            if hasattr(tok, "get_vocab"):
+++                return tok.get_vocab()
+++            return getattr(tok, "vocab", None)
+++
+++        student_vocab = _extract_vocab(tokenizer)
+++        teacher_vocab = _extract_vocab(teacher_tokenizer)
+++        if student_vocab is not None and teacher_vocab is not None:
+++            if set(student_vocab.keys()) != set(teacher_vocab.keys()):
+++                raise ValueError("Student and teacher tokenizers appear to use different vocabularies.")
+++        else:
+++            student_eos = getattr(tokenizer, "eos_token_id", None)
+++            teacher_eos = getattr(teacher_tokenizer, "eos_token_id", None)
+++            if None not in (student_eos, teacher_eos) and student_eos != teacher_eos:
+++                raise ValueError("Student and teacher tokenizers must share EOS token identifiers.")
+++
+++        if args.distill_prompts_data is None:
+++            raise ValueError("distill_prompts_data must be provided for on-policy distillation.")
+++        distill_dataset = load_prompt_only_dataset(args.distill_prompts_data, tokenizer, args)
+++        if len(distill_dataset) == 0:
+++            raise ValueError("Distillation dataset is empty.")
+++
++     print("Loading datasets")
++     train_set, valid_set, test_set = load_dataset(args, tokenizer)
++ 
++@@ -1094,7 +1334,16 @@ def run(args, training_callback: TrainingCallback = None):
++ 
++     elif args.train:
++         print("Training")
++-        train_model(args, model, tokenizer, train_set, valid_set, training_callback)
+++        train_model(
+++            args,
+++            model,
+++            tokenizer,
+++            train_set,
+++            valid_set,
+++            training_callback,
+++            teacher_model=teacher_model,
+++            distill_dataset=distill_dataset,
+++        )
++     else:
++         raise ValueError("Must provide at least one of --train or --test")
++ 
++diff --git a/mlx_lm_lora/trainer/datasets.py b/mlx_lm_lora/trainer/datasets.py
++index 8fb1ad4..79a6ba4 100644
++--- a/mlx_lm_lora/trainer/datasets.py
+++++ b/mlx_lm_lora/trainer/datasets.py
++@@ -76,12 +76,28 @@ class PromptDataset:
++         prompt_key: str = "prompt",
++     ):
++         self._data = data
++-        self.chat_key = prompt_key
+++        self.prompt_key = prompt_key
++         self.tokenizer = tokenizer
++ 
++     def process(self, d):
++-        messages = d[self.chat_key]
++-        return {"prompt": self.tokenizer.apply_chat_template(messages, add_generation_prompt=True), "prompt_text": self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)}
+++        messages = d[self.prompt_key]
+++        if isinstance(messages, str):
+++            chat = [{"role": "user", "content": messages}]
+++        elif isinstance(messages, dict):
+++            chat = [messages]
+++        else:
+++            chat = messages
+++
+++        prompt_tokens = self.tokenizer.apply_chat_template(
+++            chat,
+++            add_generation_prompt=True,
+++        )
+++        prompt_text = self.tokenizer.apply_chat_template(
+++            chat,
+++            add_generation_prompt=True,
+++            tokenize=False,
+++        )
+++        return {"prompt": prompt_tokens, "prompt_text": prompt_text}
++ 
++     def __getitem__(self, idx: int):
++         return self._data[idx]
++@@ -452,15 +468,21 @@ def create_dataset(
++                 )
++         else:
++             raise ValueError("Unsupported data format for Online DPO or CPO training.")
++-    elif train_mode in ["online_dpo", "xpo", "rlhf"]:
+++    elif train_mode in ["online_dpo", "xpo", "rlhf", "distill_on_policy"]:
++         if prompt_feature in sample:
++             return PromptDataset(
++                 data=data,
++                 tokenizer=tokenizer,
++                 prompt_key=prompt_feature,
++             )
+++        if chat_feature in sample:
+++            return PromptDataset(
+++                data=data,
+++                tokenizer=tokenizer,
+++                prompt_key=chat_feature,
+++            )
++         else:
++-            raise ValueError("Unsupported data format for Online DPO or XPO training.")
+++            raise ValueError("Unsupported data format for Online DPO, XPO, or distill_on_policy training.")
++     elif train_mode in ["grpo"]:
++         if prompt_feature in sample:
++             return GRPODataset(
++@@ -618,3 +640,33 @@ def load_dataset(args, tokenizer: PreTrainedTokenizer):
++             "Test set not found or empty. Must provide test set for evaluation."
++         )
++     return train, valid, test
+++
+++
+++def load_prompt_only_dataset(
+++    data_source: str,
+++    tokenizer: PreTrainedTokenizer,
+++    args,
+++):
+++    """
+++    Load a prompt-only dataset for on-policy distillation. Accepts either plain
+++    prompt strings (``{"prompt": ...}``) or chat-style prompts
+++    (``{"messages": [...]}``).
+++    """
+++    config = types.SimpleNamespace(
+++        train_mode="online_dpo",
+++        prompt_feature=getattr(args, "prompt_feature", "prompt"),
+++        chat_feature=getattr(args, "chat_feature", "messages"),
+++        system_feature=getattr(args, "system_feature", "system"),
+++        mask_prompt=False,
+++    )
+++
+++    data_path = Path(data_source)
+++    if data_path.exists():
+++        train, _, _ = load_local_dataset(data_path, tokenizer, config)
+++    else:
+++        train, _, _ = load_hf_dataset(data_source, tokenizer, config)
+++
+++    if len(train) == 0:
+++        raise ValueError("Distillation dataset is empty.")
+++
+++    return train
++diff --git a/mlx_lm_lora/trainer/distill_trainer.py b/mlx_lm_lora/trainer/distill_trainer.py
++new file mode 100644
++index 0000000..7660a73
++--- /dev/null
+++++ b/mlx_lm_lora/trainer/distill_trainer.py
++@@ -0,0 +1,346 @@
+++from dataclasses import dataclass, field
+++from pathlib import Path
+++from typing import List, Tuple, Optional
+++import time
+++
+++import numpy as np
+++from tqdm import tqdm
+++
+++from mlx.utils import tree_flatten
+++from mlx.nn.utils import average_gradients
+++import mlx.core as mx
+++import mlx.nn as nn
+++
+++from mlx_lm.generate import generate
+++from mlx_lm.sample_utils import make_sampler
+++from mlx_lm.tuner.callbacks import TrainingCallback
+++
+++from .datasets import CacheDataset
+++from .online_dpo_trainer import iterate_online_dpo_batches
+++from .sft_trainer import SFTTrainingArgs
+++
+++
+++@dataclass
+++class DistillationTrainingArgs(SFTTrainingArgs):
+++    max_generation_tokens: int = field(
+++        default=256,
+++        metadata={"help": "Maximum number of tokens to generate per prompt."},
+++    )
+++    distill_temperature: float = field(
+++        default=0.0,
+++        metadata={"help": "Sampling temperature for student rollouts."},
+++    )
+++
+++
+++def _sample_student_responses(
+++    model: nn.Module,
+++    tokenizer,
+++    prompt_texts: List[str],
+++    prompt_token_sequences: List[List[int]],
+++    max_tokens: int,
+++    temperature: float,
+++) -> List[List[int]]:
+++    eos_tokens = getattr(tokenizer, "eos_token_ids", None)
+++    if eos_tokens is None:
+++        single_eos = getattr(tokenizer, "eos_token_id", None)
+++        eos_tokens = [single_eos] if single_eos is not None else []
+++    eos_tokens = [tok for tok in eos_tokens if tok is not None]
+++
+++    sampler = make_sampler(
+++        temperature,
+++        top_p=1.0,
+++        min_p=0.0,
+++        min_tokens_to_keep=1,
+++        top_k=0,
+++        xtc_probability=0.0,
+++        xtc_threshold=0.0,
+++        xtc_special_tokens=tokenizer.encode("\n") + eos_tokens,
+++    )
+++
+++    completions: List[List[int]] = []
+++    for prompt_text, prompt_tokens in zip(prompt_texts, prompt_token_sequences):
+++        completion = generate(
+++            model=model,
+++            tokenizer=tokenizer,
+++            prompt=prompt_text,
+++            max_tokens=max_tokens,
+++            sampler=sampler,
+++            verbose=False,
+++        )
+++        if isinstance(completion, str):
+++            completion_ids = list(tokenizer.encode(completion))
+++        elif isinstance(completion, (list, tuple)):
+++            completion_ids = list(completion)
+++        else:
+++            completion_ids = list(completion)
+++
+++        prompt_length = len(prompt_tokens)
+++        if (
+++            prompt_length > 0
+++            and len(completion_ids) >= prompt_length
+++            and completion_ids[:prompt_length] == prompt_tokens
+++        ):
+++            completion_ids = completion_ids[prompt_length:]
+++
+++        if max_tokens is not None:
+++            completion_ids = completion_ids[:max_tokens]
+++        completions.append(list(completion_ids))
+++    return completions
+++
+++
+++def _prepare_distill_inputs(
+++    model: nn.Module,
+++    tokenizer,
+++    batch,
+++    max_tokens: int,
+++    temperature: float,
+++) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
+++    prompts, prompt_texts = batch
+++    prompt_token_sequences = []
+++    for p in prompts:
+++        if hasattr(p, "tolist"):
+++            prompt_token_sequences.append(list(p.tolist()))
+++        elif isinstance(p, (list, tuple)):
+++            prompt_token_sequences.append(list(p))
+++        else:
+++            prompt_token_sequences.append([int(p)])
+++
+++    was_training = model.training
+++    model.eval()
+++    try:
+++        completions = _sample_student_responses(
+++            model=model,
+++            tokenizer=tokenizer,
+++            prompt_texts=prompt_texts,
+++            prompt_token_sequences=prompt_token_sequences,
+++            max_tokens=max_tokens,
+++            temperature=temperature,
+++        )
+++    finally:
+++        if was_training:
+++            model.train()
+++
+++    sequences: List[List[int]] = []
+++    prompt_lengths: List[int] = []
+++    lengths: List[int] = []
+++
+++    for prompt_tokens, completion_ids in zip(prompt_token_sequences, completions):
+++        sequence = prompt_tokens + completion_ids
+++        if len(sequence) <= len(prompt_tokens):
+++            # No generated tokens; skip this sample
+++            continue
+++        sequences.append(sequence)
+++        prompt_lengths.append(len(prompt_tokens))
+++        lengths.append(len(sequence))
+++
+++    if not sequences:
+++        return None
+++
+++    max_len = max(lengths)
+++    batch_size = len(sequences)
+++    inputs = np.zeros((batch_size, max_len), dtype=np.int32)
+++    for i, seq in enumerate(sequences):
+++        inputs[i, : len(seq)] = seq
+++
+++    prompt_lengths = np.array(prompt_lengths, dtype=np.int32)
+++    lengths = np.array(lengths, dtype=np.int32)
+++    return inputs, prompt_lengths, lengths
+++
+++
+++def _compute_generation_mask(
+++    prompt_lengths: mx.array,
+++    total_lengths: mx.array,
+++    max_length: int,
+++) -> mx.array:
+++    token_positions = mx.arange(max_length - 1)[None, :]
+++    prompt_offsets = mx.maximum(prompt_lengths - 1, 0)[:, None]
+++    sequence_limits = mx.maximum(total_lengths - 1, 0)[:, None]
+++    mask = mx.logical_and(token_positions >= prompt_offsets, token_positions < sequence_limits)
+++    return mask.astype(mx.float32)
+++
+++
+++def train_on_policy_distill(
+++    model: nn.Module,
+++    teacher_model: nn.Module,
+++    tokenizer,
+++    optimizer,
+++    train_dataset: CacheDataset,
+++    args: DistillationTrainingArgs,
+++    training_callback: TrainingCallback = None,
+++):
+++    teacher_model.eval()
+++    teacher_model.freeze()
+++    model.train()
+++
+++    world = mx.distributed.init()
+++    world_size = world.size()
+++    rank = world.rank()
+++
+++    if args.gradient_accumulation_steps != 1 and rank == 0:
+++        tqdm.write(
+++            "[distill] gradient_accumulation_steps > 1 detected; overriding to 1 for distillation phase."
+++        )
+++    args.gradient_accumulation_steps = 1
+++
+++    iterator = iterate_online_dpo_batches(
+++        dataset=train_dataset,
+++        batch_size=args.batch_size,
+++        max_seq_length=args.max_seq_length,
+++        train=True,
+++    )
+++
+++    def distill_loss(model, inputs, mask):
+++        student_logits = model(inputs).astype(mx.float32)
+++        teacher_logits = mx.stop_gradient(teacher_model(inputs).astype(mx.float32))
+++
+++        student_log_probs = nn.log_softmax(student_logits[:, :-1, :], axis=-1)
+++        teacher_log_probs = nn.log_softmax(teacher_logits[:, :-1, :], axis=-1)
+++        teacher_probs = mx.exp(teacher_log_probs)
+++
+++        targets = inputs[:, 1:]
+++        kl_per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(axis=-1)
+++        kl_sum = (kl_per_token * mask).sum()
+++        num_tokens = mask.sum()
+++        if num_tokens == 0:
+++            raise ValueError("No generated tokens available for distillation loss.")
+++        loss = kl_sum / num_tokens
+++
+++        teacher_token_logps = mx.take_along_axis(
+++            teacher_log_probs, targets[..., None], axis=-1
+++        ).squeeze(-1)
+++        student_token_logps = mx.take_along_axis(
+++            student_log_probs, targets[..., None], axis=-1
+++        ).squeeze(-1)
+++
+++        teacher_sum = (teacher_token_logps * mask).sum()
+++        student_sum = (student_token_logps * mask).sum()
+++        return loss, (num_tokens, teacher_sum, student_sum, kl_sum)
+++
+++    loss_value_and_grad = nn.value_and_grad(model, distill_loss)
+++
+++    loss_numerator = mx.array(0.0, dtype=mx.float32)
+++    total_tokens = mx.array(0.0, dtype=mx.float32)
+++    kl_total = mx.array(0.0, dtype=mx.float32)
+++    teacher_log_total = mx.array(0.0, dtype=mx.float32)
+++    student_log_total = mx.array(0.0, dtype=mx.float32)
+++
+++    window_tokens = mx.array(0.0, dtype=mx.float32)
+++    window_time = 0.0
+++
+++    pbar = tqdm(total=args.iters, desc="Distill", disable=rank != 0)
+++    it = 0
+++    tic = time.perf_counter()
+++    consecutive_empty = 0
+++    max_empty = 5
+++    while it < args.iters:
+++        batch = next(iterator)
+++        prepared = _prepare_distill_inputs(
+++            model=model,
+++            tokenizer=tokenizer,
+++            batch=batch,
+++            max_tokens=args.max_generation_tokens,
+++            temperature=args.distill_temperature,
+++        )
+++        if prepared is None:
+++            consecutive_empty += 1
+++            if consecutive_empty >= max_empty:
+++                raise RuntimeError(
+++                    "Failed to generate non-empty student completions after multiple attempts."
+++                )
+++            tic = time.perf_counter()
+++            continue
+++        consecutive_empty = 0
+++
+++        inputs_np, prompt_lengths_np, lengths_np = prepared
+++        inputs = mx.array(inputs_np)
+++        prompt_lengths = mx.array(prompt_lengths_np)
+++        lengths = mx.array(lengths_np)
+++        mask = _compute_generation_mask(prompt_lengths, lengths, inputs.shape[1])
+++
+++        (loss_value, metrics), grad = loss_value_and_grad(model, inputs, mask)
+++        grad = average_gradients(grad)
+++        optimizer.update(model, grad)
+++
+++        num_tokens = metrics[0]
+++        teacher_sum = metrics[1]
+++        student_sum = metrics[2]
+++        kl_sum = metrics[3]
+++
+++        loss_numerator += loss_value * num_tokens
+++        total_tokens += num_tokens
+++        kl_total += kl_sum
+++        teacher_log_total += teacher_sum
+++        student_log_total += student_sum
+++        window_tokens += num_tokens
+++
+++        it += 1
+++        pbar.update(1)
+++        window_time += time.perf_counter() - tic
+++
+++        if it % args.steps_per_report == 0 or it == args.iters:
+++            mx.eval(loss_numerator, total_tokens, kl_total, teacher_log_total, student_log_total, window_tokens)
+++
+++            total_loss = mx.distributed.all_sum(loss_numerator, stream=mx.cpu).item()
+++            total_tok = mx.distributed.all_sum(total_tokens, stream=mx.cpu).item()
+++            total_kl = mx.distributed.all_sum(kl_total, stream=mx.cpu).item()
+++            total_teacher = mx.distributed.all_sum(teacher_log_total, stream=mx.cpu).item()
+++            total_student = mx.distributed.all_sum(student_log_total, stream=mx.cpu).item()
+++            window_tok = mx.distributed.all_sum(window_tokens, stream=mx.cpu).item()
+++
+++            avg_loss = total_loss / max(total_tok, 1.0)
+++            avg_kl = total_kl / max(total_tok, 1.0)
+++            avg_teacher = total_teacher / max(total_tok, 1.0)
+++            avg_student = total_student / max(total_tok, 1.0)
+++            avg_tokens_per_sec = window_tok / max(window_time, 1e-6)
+++            lr_value = optimizer.learning_rate.item()
+++
+++            if rank == 0:
+++                tqdm.write(
+++                    f"\nDistill iter {it}: "
+++                    f"loss {avg_loss:.4f}, "
+++                    f"kl/token {avg_kl:.4f}, "
+++                    f"teacher_logp {avg_teacher:.4f}, "
+++                    f"student_logp {avg_student:.4f}, "
+++                    f"lr {lr_value:.3e}, "
+++                    f"window_tok/s {avg_tokens_per_sec:.3f}"
+++                )
+++
+++            if training_callback is not None:
+++                train_info = {
+++                    "iteration": it,
+++                    "train_loss": avg_loss,
+++                    "learning_rate": lr_value,
+++                    "tokens_per_second": avg_tokens_per_sec,
+++                    "kl_per_token": avg_kl,
+++                    "teacher_logp": avg_teacher,
+++                    "student_logp": avg_student,
+++                }
+++                training_callback.on_train_loss_report(train_info)
+++
+++            loss_numerator = mx.array(0.0, dtype=mx.float32)
+++            total_tokens = mx.array(0.0, dtype=mx.float32)
+++            kl_total = mx.array(0.0, dtype=mx.float32)
+++            teacher_log_total = mx.array(0.0, dtype=mx.float32)
+++            student_log_total = mx.array(0.0, dtype=mx.float32)
+++            window_tokens = mx.array(0.0, dtype=mx.float32)
+++            window_time = 0.0
+++
+++        if it % args.steps_per_save == 0 and rank == 0:
+++            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
+++            mx.save_safetensors(str(args.adapter_file), adapter_weights)
+++            checkpoint = (
+++                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
+++            )
+++            mx.save_safetensors(str(checkpoint), adapter_weights)
+++            tqdm.write(
+++                f"\n"
+++                f"Distill iter {it}: Saved adapter weights to "
+++                f"{args.adapter_file} and {checkpoint}."
+++            )
+++
+++        tic = time.perf_counter()
+++
+++    if rank == 0:
+++        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
+++        mx.save_safetensors(str(args.adapter_file), adapter_weights)
+++        tqdm.write(f"Saved final weights to {args.adapter_file}.")
+diff --git a/mlx_lm_lora/train.py b/mlx_lm_lora/train.py
+index 9efa33a..9ea2a86 100644
+--- a/mlx_lm_lora/train.py
++++ b/mlx_lm_lora/train.py
+@@ -2,9 +2,9 @@ from pathlib import Path
+ import importlib.util
+ import argparse
+ import math
+-import yaml
+ import sys
+ import re
++import yaml
+ 
+ import numpy as np
+ 
+@@ -26,7 +26,8 @@ from .trainer.rflhf_trainer import RLHFTrainingArgs, evaluate_rlhf, train_rlhf
+ from .trainer.xpo_trainer import  XPOTrainingArgs, evaluate_xpo, train_xpo
+ from .trainer.dpo_trainer import DPOTrainingArgs, evaluate_dpo, train_dpo
+ from .trainer.cpo_trainer import CPOTrainingArgs, evaluate_cpo, train_cpo
+-from .trainer.datasets import CacheDataset, load_dataset
++from .trainer.distill_trainer import DistillationTrainingArgs, train_on_policy_distill
++from .trainer.datasets import CacheDataset, load_dataset, load_prompt_only_dataset
+ from .utils import fuse_and_save_model, from_pretrained
+ 
+ from mlx_lm.tuner.utils import (
+@@ -70,6 +71,7 @@ CONFIG_DEFAULTS = {
+         "adafactor": {},
+     },
+     "data": "data/",
++    "distill_prompts_data": None,
+     "seed": 0,
+     "num_layers": 16,
+     "batch_size": 4,
+@@ -86,9 +88,11 @@ CONFIG_DEFAULTS = {
+     "test": False,
+     "test_batches": 500,
+     "max_seq_length": 2048,
++    "max_generation_tokens": 256,
+     "config": None,
+     "grad_checkpoint": False,
+     "lr_schedule": None,
++    "training_schedule": None,
+     "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 10.0},
+     "mask_prompt": False,
+     "fuse": True,
+@@ -106,6 +110,8 @@ CONFIG_DEFAULTS = {
+     "judge": None,
+     "judge_config": {},
+     "alpha": 1e-5,
++    "teacher_model": None,
++    "distill_temperature": 0.0,
+ 
+     # GRPO args
+     "group_size": 4,
+@@ -147,6 +153,65 @@ def calculate_iters(train_set, batch_size, epochs) -> int:
+     return iters
+ 
+ 
++def parse_training_schedule(schedule_str: str):
++    if schedule_str is None:
++        return None
++    entries = []
++    for raw_part in schedule_str.split(","):
++        part = raw_part.strip()
++        if not part:
++            continue
++        if ":" not in part:
++            raise ValueError(
++                f"Invalid training schedule entry '{part}'. Expected format mode:weight."
++            )
++        mode, weight = part.split(":", 1)
++        try:
++            weight_val = float(weight.strip())
++        except ValueError as exc:
++            raise ValueError(f"Invalid weight '{weight}' in training schedule.") from exc
++        entries.append({"mode": mode.strip(), "weight": weight_val})
++    if not entries:
++        raise ValueError("Training schedule was provided but no valid entries were parsed.")
++    if any(item["weight"] <= 0 for item in entries):
++        raise ValueError("Training schedule weights must be positive.")
++    return entries
++
++
++def allocate_schedule_iterations(total_iters: int, entries):
++    if total_iters is None:
++        raise ValueError("Total iterations must be specified when using a training schedule.")
++    total_weight = sum(item["weight"] for item in entries)
++    if total_weight <= 0:
++        raise ValueError("Training schedule weights must sum to a positive value.")
++
++    exact_counts = [
++        (item, total_iters * item["weight"] / total_weight) for item in entries
++    ]
++    assigned_counts = []
++    residuals = []
++    for item, exact in exact_counts:
++        count = int(math.floor(exact))
++        assigned_counts.append([item, count])
++        residuals.append((exact - count, item))
++
++    assigned_total = sum(count for _, count in assigned_counts)
++    remaining = total_iters - assigned_total
++    residuals.sort(key=lambda tup: tup[0], reverse=True)
++
++    idx = 0
++    while remaining > 0 and residuals:
++        _, item = residuals[idx % len(residuals)]
++        for entry in assigned_counts:
++            if entry[0] is item:
++                entry[1] += 1
++                remaining -= 1
++                break
++        idx += 1
++
++    return [(entry["mode"], count) for entry, count in assigned_counts]
++
++
+ def build_parser():
+     parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
+     parser.add_argument(
+@@ -154,6 +219,12 @@ def build_parser():
+         type=str,
+         help="The path to the local model directory or Hugging Face repo.",
+     )
++    parser.add_argument(
++        "--teacher-model",
++        type=str,
++        default=None,
++        help="Optional teacher model path for on-policy distillation.",
++    )
+     parser.add_argument(
+         "--load-in-4bits",
+         action="store_true",
+@@ -188,6 +259,12 @@ def build_parser():
+             "of a Hugging Face dataset (e.g., 'mlx-community/wikisql')"
+         ),
+     )
++    parser.add_argument(
++        "--distill-prompts-data",
++        type=str,
++        default=None,
++        help="Directory or dataset id providing prompts for on-policy distillation.",
++    )
+     parser.add_argument(
+         "--train-type",
+         type=str,
+@@ -198,8 +275,14 @@ def build_parser():
+         "--train-mode",
+         type=str,
+         default="sft",
+-        choices=["sft", "dpo", "cpo", "orpo", "grpo", "online_dpo", "xpo", "rlhf"],
+-        help="Training mode: sft, dpo, rlhf, online_dpo, xpo, cpo, orpo, or grpo, default is sft",
++        choices=["sft", "dpo", "cpo", "orpo", "grpo", "online_dpo", "xpo", "rlhf", "distill_on_policy"],
++        help="Training mode: sft, dpo, rlhf, online_dpo, xpo, cpo, orpo, grpo, or distill_on_policy. Default is sft",
++    )
++    parser.add_argument(
++        "--training-schedule",
++        type=str,
++        default=None,
++        help="Comma-separated list of mode:weight pairs to sequence training phases.",
+     )
+     parser.add_argument(
+         "--optimizer",
+@@ -295,6 +378,18 @@ def build_parser():
+         type=int,
+         help="Maximum sequence length.",
+     )
++    parser.add_argument(
++        "--max-generation-tokens",
++        type=int,
++        default=256,
++        help="Maximum number of tokens to generate during on-policy distillation.",
++    )
++    parser.add_argument(
++        "--distill-temperature",
++        type=float,
++        default=0.0,
++        help="Sampling temperature when generating student rollouts for distillation.",
++    )
+     parser.add_argument(
+         "-c",
+         "--config",
+@@ -455,6 +550,8 @@ def train_model(
+     train_set,
+     valid_set,
+     training_callback: TrainingCallback = None,
++    teacher_model: nn.Module = None,
++    distill_dataset=None,
+ ):
+     mx.random.seed(args.seed)
+ 
+@@ -567,6 +664,71 @@ def train_model(
+ 
+     opt = opt_class(learning_rate=lr, **optimizer_config)
+ 
++    schedule_entries = getattr(args, "_parsed_training_schedule", None)
++    if schedule_entries:
++        iteration_plan = allocate_schedule_iterations(args.iters, schedule_entries)
++        sft_train_cache = CacheDataset(train_set) if train_set and len(train_set) else None
++        sft_valid_cache = CacheDataset(valid_set) if valid_set and len(valid_set) else None
++        distill_cache = CacheDataset(distill_dataset) if distill_dataset and len(distill_dataset) else None
++
++        for mode, count in iteration_plan:
++            if count <= 0:
++                continue
++            if mode == "sft":
++                if sft_train_cache is None or sft_valid_cache is None:
++                    raise ValueError("SFT dataset is required but not available for the training schedule.")
++                sft_training_args = SFTTrainingArgs(
++                    batch_size=args.batch_size,
++                    iters=count,
++                    val_batches=args.val_batches,
++                    steps_per_report=args.steps_per_report,
++                    steps_per_eval=args.steps_per_eval,
++                    steps_per_save=args.save_every,
++                    adapter_file=adapter_file,
++                    max_seq_length=args.max_seq_length,
++                    grad_checkpoint=args.grad_checkpoint,
++                    gradient_accumulation_steps=args.gradient_accumulation_steps,
++                )
++                train_sft(
++                    model=model,
++                    args=sft_training_args,
++                    optimizer=opt,
++                    train_dataset=sft_train_cache,
++                    val_dataset=sft_valid_cache,
++                    training_callback=training_callback,
++                )
++            elif mode == "distill_on_policy":
++                if teacher_model is None:
++                    raise ValueError("Teacher model required for distillation in training schedule.")
++                if distill_cache is None:
++                    raise ValueError("Distillation dataset required for training schedule entry.")
++                distill_args = DistillationTrainingArgs(
++                    batch_size=args.batch_size,
++                    iters=count,
++                    val_batches=0,
++                    steps_per_report=args.steps_per_report,
++                    steps_per_eval=args.steps_per_eval,
++                    steps_per_save=args.save_every,
++                    adapter_file=adapter_file,
++                    max_seq_length=args.max_seq_length,
++                    grad_checkpoint=False,
++                    gradient_accumulation_steps=args.gradient_accumulation_steps,
++                    max_generation_tokens=args.max_generation_tokens,
++                    distill_temperature=args.distill_temperature,
++                )
++                train_on_policy_distill(
++                    model=model,
++                    teacher_model=teacher_model,
++                    tokenizer=tokenizer,
++                    optimizer=opt,
++                    train_dataset=distill_cache,
++                    args=distill_args,
++                    training_callback=training_callback,
++                )
++            else:
++                raise ValueError(f"Unsupported mode '{mode}' in training schedule.")
++        return
++
+     if args.train_mode == "orpo":
+         orpo_training_args = ORPOTrainingArgs(
+             batch_size=args.batch_size,
+@@ -864,6 +1026,35 @@ def train_model(
+             training_callback=training_callback,
+         )
+ 
++    elif args.train_mode == "distill_on_policy":
++        if teacher_model is None:
++            raise ValueError("Teacher model must be provided for distill_on_policy training.")
++        if distill_dataset is None or len(distill_dataset) == 0:
++            raise ValueError("A non-empty distillation dataset is required for distill_on_policy training.")
++        distill_args = DistillationTrainingArgs(
++            batch_size=args.batch_size,
++            iters=args.iters,
++            val_batches=0,
++            steps_per_report=args.steps_per_report,
++            steps_per_eval=args.steps_per_eval,
++            steps_per_save=args.save_every,
++            adapter_file=adapter_file,
++            max_seq_length=args.max_seq_length,
++            grad_checkpoint=False,
++            gradient_accumulation_steps=args.gradient_accumulation_steps,
++            max_generation_tokens=args.max_generation_tokens,
++            distill_temperature=args.distill_temperature,
++        )
++        train_on_policy_distill(
++            model=model,
++            teacher_model=teacher_model,
++            tokenizer=tokenizer,
++            optimizer=opt,
++            train_dataset=CacheDataset(distill_dataset),
++            args=distill_args,
++            training_callback=training_callback,
++        )
++
+     elif args.train_mode == "sft":
+         sft_training_args = SFTTrainingArgs(
+             batch_size=args.batch_size,
+@@ -935,6 +1126,9 @@ def evaluate_model(args, model: nn.Module, tokenizer, test_set):
+         for metric_name, metric_value in test_metrics.items():
+             print(f"  {metric_name}: {float(metric_value):.3f}")
+ 
++    elif args.train_mode == "distill_on_policy":
++        print("Evaluation for distill_on_policy training is not currently implemented.")
++
+     elif args.train_mode == "rlhf":
+         if args.reference_model_path:
+             reference_model, _ = load(args.reference_model_path)
+@@ -1072,21 +1266,91 @@ def run(args, training_callback: TrainingCallback = None):
+     # model, tokenizer = load(args.model)
+ 
+     if args.load_in_4bits:
+-        quanziation_config = {"bits": 4, "group_size": 64}
++        quantization_config = {"bits": 4, "group_size": 64}
+     elif args.load_in_6bits:
+-        quanziation_config = {"bits": 6, "group_size": 64}
++        quantization_config = {"bits": 6, "group_size": 64}
+     elif args.load_in_8bits:
+-        quanziation_config = {"bits": 8, "group_size": 64}
++        quantization_config = {"bits": 8, "group_size": 64}
+     else:
+-        quanziation_config = None
++        quantization_config = None
+ 
+     model, tokenizer = from_pretrained(
+         model=args.model,
+-        quantized_load=quanziation_config,
++        quantized_load=quantization_config,
+     )
+ 
+-    print("Loading datasets")
+-    train_set, valid_set, test_set = load_dataset(args, tokenizer)
++    schedule_entries = None
++    raw_schedule = getattr(args, "training_schedule", None)
++    if raw_schedule:
++        if isinstance(raw_schedule, str):
++            schedule_entries = parse_training_schedule(raw_schedule)
++        elif isinstance(raw_schedule, list):
++            schedule_entries = raw_schedule
++        else:
++            raise ValueError("training_schedule must be a string or list of schedule entries.")
++    setattr(args, "_parsed_training_schedule", schedule_entries)
++
++    requires_distill = args.train_mode == "distill_on_policy"
++    if schedule_entries:
++        requires_distill = requires_distill or any(
++            entry["mode"] == "distill_on_policy" for entry in schedule_entries
++        )
++
++    teacher_model = None
++    distill_dataset = None
++    if args.train and requires_distill:
++        if args.teacher_model is None:
++            raise ValueError("A teacher model must be provided when using on-policy distillation.")
++        teacher_model, teacher_tokenizer = from_pretrained(args.teacher_model)
++
++        def _extract_vocab(tok):
++            if hasattr(tok, "get_vocab"):
++                return tok.get_vocab()
++            return getattr(tok, "vocab", None)
++
++        student_vocab = _extract_vocab(tokenizer)
++        teacher_vocab = _extract_vocab(teacher_tokenizer)
++        if student_vocab is not None and teacher_vocab is not None:
++            if student_vocab != teacher_vocab:
++                raise ValueError("Student and teacher tokenizers appear to use different token-id mappings.")
++        else:
++            special_tokens = [
++                "bos_token_id",
++                "eos_token_id",
++                "pad_token_id",
++                "unk_token_id",
++                "sep_token_id",
++            ]
++            for attr in special_tokens:
++                student_id = getattr(tokenizer, attr, None)
++                teacher_id = getattr(teacher_tokenizer, attr, None)
++                if None not in (student_id, teacher_id) and student_id != teacher_id:
++                    raise ValueError(
++                        f"Student and teacher tokenizers must agree on {attr.replace('_', ' ')}."
++                    )
++
++        if args.distill_prompts_data is None:
++            raise ValueError("distill_prompts_data must be provided for on-policy distillation.")
++        distill_dataset = load_prompt_only_dataset(args.distill_prompts_data, tokenizer, args)
++        if len(distill_dataset) == 0:
++            raise ValueError("Distillation dataset is empty.")
++
++    need_train_data = True
++    if args.train and not args.test:
++        if schedule_entries:
++            schedule_modes = {entry["mode"] for entry in schedule_entries}
++            if schedule_modes <= {"distill_on_policy"}:
++                need_train_data = False
++        elif args.train_mode == "distill_on_policy":
++            need_train_data = False
++
++    if need_train_data or args.test:
++        print("Loading datasets")
++        train_set, valid_set, test_set = load_dataset(args, tokenizer)
++    else:
++        train_set = []
++        valid_set = []
++        test_set = []
+ 
+     if args.test and not args.train:
+         if args.adapter_path != "":
+@@ -1094,7 +1358,16 @@ def run(args, training_callback: TrainingCallback = None):
+ 
+     elif args.train:
+         print("Training")
+-        train_model(args, model, tokenizer, train_set, valid_set, training_callback)
++        train_model(
++            args,
++            model,
++            tokenizer,
++            train_set,
++            valid_set,
++            training_callback,
++            teacher_model=teacher_model,
++            distill_dataset=distill_dataset,
++        )
+     else:
+         raise ValueError("Must provide at least one of --train or --test")
+ 
+diff --git a/mlx_lm_lora/trainer/datasets.py b/mlx_lm_lora/trainer/datasets.py
+index 8fb1ad4..79a6ba4 100644
+--- a/mlx_lm_lora/trainer/datasets.py
++++ b/mlx_lm_lora/trainer/datasets.py
+@@ -76,12 +76,28 @@ class PromptDataset:
+         prompt_key: str = "prompt",
+     ):
+         self._data = data
+-        self.chat_key = prompt_key
++        self.prompt_key = prompt_key
+         self.tokenizer = tokenizer
+ 
+     def process(self, d):
+-        messages = d[self.chat_key]
+-        return {"prompt": self.tokenizer.apply_chat_template(messages, add_generation_prompt=True), "prompt_text": self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)}
++        messages = d[self.prompt_key]
++        if isinstance(messages, str):
++            chat = [{"role": "user", "content": messages}]
++        elif isinstance(messages, dict):
++            chat = [messages]
++        else:
++            chat = messages
++
++        prompt_tokens = self.tokenizer.apply_chat_template(
++            chat,
++            add_generation_prompt=True,
++        )
++        prompt_text = self.tokenizer.apply_chat_template(
++            chat,
++            add_generation_prompt=True,
++            tokenize=False,
++        )
++        return {"prompt": prompt_tokens, "prompt_text": prompt_text}
+ 
+     def __getitem__(self, idx: int):
+         return self._data[idx]
+@@ -452,15 +468,21 @@ def create_dataset(
+                 )
+         else:
+             raise ValueError("Unsupported data format for Online DPO or CPO training.")
+-    elif train_mode in ["online_dpo", "xpo", "rlhf"]:
++    elif train_mode in ["online_dpo", "xpo", "rlhf", "distill_on_policy"]:
+         if prompt_feature in sample:
+             return PromptDataset(
+                 data=data,
+                 tokenizer=tokenizer,
+                 prompt_key=prompt_feature,
+             )
++        if chat_feature in sample:
++            return PromptDataset(
++                data=data,
++                tokenizer=tokenizer,
++                prompt_key=chat_feature,
++            )
+         else:
+-            raise ValueError("Unsupported data format for Online DPO or XPO training.")
++            raise ValueError("Unsupported data format for Online DPO, XPO, or distill_on_policy training.")
+     elif train_mode in ["grpo"]:
+         if prompt_feature in sample:
+             return GRPODataset(
+@@ -618,3 +640,33 @@ def load_dataset(args, tokenizer: PreTrainedTokenizer):
+             "Test set not found or empty. Must provide test set for evaluation."
+         )
+     return train, valid, test
++
++
++def load_prompt_only_dataset(
++    data_source: str,
++    tokenizer: PreTrainedTokenizer,
++    args,
++):
++    """
++    Load a prompt-only dataset for on-policy distillation. Accepts either plain
++    prompt strings (``{"prompt": ...}``) or chat-style prompts
++    (``{"messages": [...]}``).
++    """
++    config = types.SimpleNamespace(
++        train_mode="online_dpo",
++        prompt_feature=getattr(args, "prompt_feature", "prompt"),
++        chat_feature=getattr(args, "chat_feature", "messages"),
++        system_feature=getattr(args, "system_feature", "system"),
++        mask_prompt=False,
++    )
++
++    data_path = Path(data_source)
++    if data_path.exists():
++        train, _, _ = load_local_dataset(data_path, tokenizer, config)
++    else:
++        train, _, _ = load_hf_dataset(data_source, tokenizer, config)
++
++    if len(train) == 0:
++        raise ValueError("Distillation dataset is empty.")
++
++    return train
+diff --git a/mlx_lm_lora/trainer/distill_trainer.py b/mlx_lm_lora/trainer/distill_trainer.py
+new file mode 100644
+index 0000000..d457602
+--- /dev/null
++++ b/mlx_lm_lora/trainer/distill_trainer.py
+@@ -0,0 +1,364 @@
++from dataclasses import dataclass, field
++from pathlib import Path
++from typing import List, Tuple, Optional
++import time
++
++import numpy as np
++from tqdm import tqdm
++
++from mlx.utils import tree_flatten
++from mlx.nn.utils import average_gradients
++import mlx.core as mx
++import mlx.nn as nn
++
++from mlx_lm.generate import generate
++from mlx_lm.sample_utils import make_sampler
++from mlx_lm.tuner.callbacks import TrainingCallback
++
++from .datasets import CacheDataset
++from .online_dpo_trainer import iterate_online_dpo_batches
++from .sft_trainer import SFTTrainingArgs
++
++
++@dataclass
++class DistillationTrainingArgs(SFTTrainingArgs):
++    max_generation_tokens: int = field(
++        default=256,
++        metadata={"help": "Maximum number of tokens to generate per prompt."},
++    )
++    distill_temperature: float = field(
++        default=0.0,
++        metadata={"help": "Sampling temperature for student rollouts."},
++    )
++
++
++def _sample_student_responses(
++    model: nn.Module,
++    tokenizer,
++    prompt_texts: List[str],
++    prompt_token_sequences: List[List[int]],
++    max_tokens: int,
++    temperature: float,
++    max_output_tokens: List[int],
++) -> List[List[int]]:
++    eos_tokens = getattr(tokenizer, "eos_token_ids", None)
++    if eos_tokens is None:
++        single_eos = getattr(tokenizer, "eos_token_id", None)
++        eos_tokens = [single_eos] if single_eos is not None else []
++    eos_tokens = [tok for tok in eos_tokens if tok is not None]
++
++    sampler = make_sampler(
++        temperature,
++        top_p=1.0,
++        min_p=0.0,
++        min_tokens_to_keep=1,
++        top_k=0,
++        xtc_probability=0.0,
++        xtc_threshold=0.0,
++        xtc_special_tokens=tokenizer.encode("\n") + eos_tokens,
++    )
++
++    completions: List[List[int]] = []
++    for prompt_text, prompt_tokens, allowed_tokens in zip(
++        prompt_texts, prompt_token_sequences, max_output_tokens
++    ):
++        current_max = max(1, min(max_tokens, allowed_tokens))
++        completion = generate(
++            model=model,
++            tokenizer=tokenizer,
++            prompt=prompt_text,
++            max_tokens=current_max,
++            sampler=sampler,
++            verbose=False,
++        )
++        if isinstance(completion, str):
++            completion_ids = list(tokenizer.encode(completion))
++        elif isinstance(completion, (list, tuple)):
++            completion_ids = list(completion)
++        else:
++            completion_ids = list(completion)
++
++        prompt_length = len(prompt_tokens)
++        if (
++            prompt_length > 0
++            and len(completion_ids) >= prompt_length
++            and completion_ids[:prompt_length] == prompt_tokens
++        ):
++            completion_ids = completion_ids[prompt_length:]
++
++        if current_max is not None:
++            completion_ids = completion_ids[:current_max]
++        completions.append(list(completion_ids))
++    return completions
++
++
++def _prepare_distill_inputs(
++    model: nn.Module,
++    tokenizer,
++    batch,
++    max_tokens: int,
++    temperature: float,
++    max_seq_length: int,
++) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
++    prompts, prompt_texts = batch
++    prompt_token_sequences = []
++    for p in prompts:
++        if hasattr(p, "tolist"):
++            prompt_token_sequences.append(list(p.tolist()))
++        elif isinstance(p, (list, tuple)):
++            prompt_token_sequences.append(list(p))
++        else:
++            prompt_token_sequences.append([int(p)])
++
++    allowed_token_budget: List[int] = []
++    for prompt_tokens in prompt_token_sequences:
++        headroom = max_seq_length - len(prompt_tokens) - 1
++        allowed_token_budget.append(max(1, headroom))
++
++    was_training = model.training
++    model.eval()
++    try:
++        completions = _sample_student_responses(
++            model=model,
++            tokenizer=tokenizer,
++            prompt_texts=prompt_texts,
++            prompt_token_sequences=prompt_token_sequences,
++            max_tokens=max_tokens,
++            temperature=temperature,
++            max_output_tokens=allowed_token_budget,
++        )
++    finally:
++        if was_training:
++            model.train()
++
++    sequences: List[List[int]] = []
++    prompt_lengths: List[int] = []
++    lengths: List[int] = []
++
++    for prompt_tokens, completion_ids in zip(prompt_token_sequences, completions):
++        allowed_total = max_seq_length - len(prompt_tokens)
++        trimmed_completion = completion_ids[:max(0, allowed_total)]
++        sequence = prompt_tokens + trimmed_completion
++        if len(sequence) <= len(prompt_tokens):
++            # No generated tokens; skip this sample
++            continue
++        sequences.append(sequence)
++        prompt_lengths.append(len(prompt_tokens))
++        lengths.append(len(sequence))
++
++    if not sequences:
++        return None
++
++    max_len = max(lengths)
++    batch_size = len(sequences)
++    inputs = np.zeros((batch_size, max_len), dtype=np.int32)
++    for i, seq in enumerate(sequences):
++        inputs[i, : len(seq)] = seq
++
++    prompt_lengths = np.array(prompt_lengths, dtype=np.int32)
++    lengths = np.array(lengths, dtype=np.int32)
++    return inputs, prompt_lengths, lengths
++
++
++def _compute_generation_mask(
++    prompt_lengths: mx.array,
++    total_lengths: mx.array,
++    max_length: int,
++) -> mx.array:
++    token_positions = mx.arange(max_length - 1)[None, :]
++    prompt_offsets = mx.maximum(prompt_lengths - 1, 0)[:, None]
++    sequence_limits = mx.maximum(total_lengths - 1, 0)[:, None]
++    mask = mx.logical_and(token_positions >= prompt_offsets, token_positions < sequence_limits)
++    return mask.astype(mx.float32)
++
++
++def train_on_policy_distill(
++    model: nn.Module,
++    teacher_model: nn.Module,
++    tokenizer,
++    optimizer,
++    train_dataset: CacheDataset,
++    args: DistillationTrainingArgs,
++    training_callback: TrainingCallback = None,
++):
++    teacher_model.eval()
++    if hasattr(teacher_model, "freeze"):
++        teacher_model.freeze()
++    elif hasattr(teacher_model, "parameters"):
++        for param in teacher_model.parameters():
++            if hasattr(param, "requires_grad"):
++                param.requires_grad = False
++    model.train()
++
++    world = mx.distributed.init()
++    rank = world.rank()
++
++    if args.gradient_accumulation_steps != 1 and rank == 0:
++        tqdm.write(
++            "[distill] gradient_accumulation_steps > 1 detected; overriding to 1 for distillation phase."
++        )
++    grad_accum_steps = 1
++
++    iterator = iterate_online_dpo_batches(
++        dataset=train_dataset,
++        batch_size=args.batch_size,
++        max_seq_length=args.max_seq_length,
++        train=True,
++    )
++
++    def distill_loss(model, inputs, mask):
++        student_logits = model(inputs).astype(mx.float32)
++        teacher_logits = mx.stop_gradient(teacher_model(inputs).astype(mx.float32))
++
++        student_log_probs = nn.log_softmax(student_logits[:, :-1, :], axis=-1)
++        teacher_log_probs = nn.log_softmax(teacher_logits[:, :-1, :], axis=-1)
++        teacher_probs = mx.exp(teacher_log_probs)
++
++        targets = inputs[:, 1:]
++        kl_per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(axis=-1)
++        kl_sum = (kl_per_token * mask).sum()
++        num_tokens = mask.sum()
++        if num_tokens == 0:
++            raise ValueError("No generated tokens available for distillation loss.")
++        loss = kl_sum / num_tokens
++
++        teacher_token_logps = mx.take_along_axis(
++            teacher_log_probs, targets[..., None], axis=-1
++        ).squeeze(-1)
++        student_token_logps = mx.take_along_axis(
++            student_log_probs, targets[..., None], axis=-1
++        ).squeeze(-1)
++
++        teacher_sum = (teacher_token_logps * mask).sum()
++        student_sum = (student_token_logps * mask).sum()
++        return loss, (num_tokens, teacher_sum, student_sum, kl_sum)
++
++    loss_value_and_grad = nn.value_and_grad(model, distill_loss)
++
++    loss_numerator = mx.array(0.0, dtype=mx.float32)
++    total_tokens = mx.array(0.0, dtype=mx.float32)
++    kl_total = mx.array(0.0, dtype=mx.float32)
++    teacher_log_total = mx.array(0.0, dtype=mx.float32)
++    student_log_total = mx.array(0.0, dtype=mx.float32)
++
++    window_tokens = mx.array(0.0, dtype=mx.float32)
++    window_time = 0.0
++
++    pbar = tqdm(total=args.iters, desc="Distill", disable=rank != 0)
++    it = 0
++    tic = time.perf_counter()
++    consecutive_empty = 0
++    max_empty = 5
++    while it < args.iters:
++        batch = next(iterator)
++        prepared = _prepare_distill_inputs(
++            model=model,
++            tokenizer=tokenizer,
++            batch=batch,
++            max_tokens=args.max_generation_tokens,
++            temperature=args.distill_temperature,
++            max_seq_length=args.max_seq_length,
++        )
++        if prepared is None:
++            consecutive_empty += 1
++            if consecutive_empty >= max_empty:
++                raise RuntimeError(
++                    "Failed to generate non-empty student completions after multiple attempts."
++                )
++            tic = time.perf_counter()
++            continue
++        consecutive_empty = 0
++
++        inputs_np, prompt_lengths_np, lengths_np = prepared
++        inputs = mx.array(inputs_np)
++        prompt_lengths = mx.array(prompt_lengths_np)
++        lengths = mx.array(lengths_np)
++        mask = _compute_generation_mask(prompt_lengths, lengths, inputs.shape[1])
++
++        (loss_value, metrics), grad = loss_value_and_grad(model, inputs, mask)
++        grad = average_gradients(grad)
++        optimizer.update(model, grad)
++
++        num_tokens = metrics[0]
++        teacher_sum = metrics[1]
++        student_sum = metrics[2]
++        kl_sum = metrics[3]
++
++        loss_numerator += loss_value * num_tokens
++        total_tokens += num_tokens
++        kl_total += kl_sum
++        teacher_log_total += teacher_sum
++        student_log_total += student_sum
++        window_tokens += num_tokens
++
++        it += 1
++        pbar.update(1)
++        window_time += time.perf_counter() - tic
++
++        if it % args.steps_per_report == 0 or it == args.iters:
++            mx.eval(loss_numerator, total_tokens, kl_total, teacher_log_total, student_log_total, window_tokens)
++
++            total_loss = mx.distributed.all_sum(loss_numerator, stream=mx.cpu).item()
++            total_tok = mx.distributed.all_sum(total_tokens, stream=mx.cpu).item()
++            total_kl = mx.distributed.all_sum(kl_total, stream=mx.cpu).item()
++            total_teacher = mx.distributed.all_sum(teacher_log_total, stream=mx.cpu).item()
++            total_student = mx.distributed.all_sum(student_log_total, stream=mx.cpu).item()
++            window_tok = mx.distributed.all_sum(window_tokens, stream=mx.cpu).item()
++
++            avg_loss = total_loss / max(total_tok, 1.0)
++            avg_kl = total_kl / max(total_tok, 1.0)
++            avg_teacher = total_teacher / max(total_tok, 1.0)
++            avg_student = total_student / max(total_tok, 1.0)
++            avg_tokens_per_sec = window_tok / max(window_time, 1e-6)
++            lr_value = optimizer.learning_rate.item()
++
++            if rank == 0:
++                tqdm.write(
++                    f"\nDistill iter {it}: "
++                    f"loss {avg_loss:.4f}, "
++                    f"kl/token {avg_kl:.4f}, "
++                    f"teacher_logp {avg_teacher:.4f}, "
++                    f"student_logp {avg_student:.4f}, "
++                    f"lr {lr_value:.3e}, "
++                    f"window_tok/s {avg_tokens_per_sec:.3f}"
++                )
++
++            if training_callback is not None:
++                train_info = {
++                    "iteration": it,
++                    "train_loss": avg_loss,
++                    "learning_rate": lr_value,
++                    "tokens_per_second": avg_tokens_per_sec,
++                    "kl_per_token": avg_kl,
++                    "teacher_logp": avg_teacher,
++                    "student_logp": avg_student,
++                }
++                training_callback.on_train_loss_report(train_info)
++
++            loss_numerator = mx.array(0.0, dtype=mx.float32)
++            total_tokens = mx.array(0.0, dtype=mx.float32)
++            kl_total = mx.array(0.0, dtype=mx.float32)
++            teacher_log_total = mx.array(0.0, dtype=mx.float32)
++            student_log_total = mx.array(0.0, dtype=mx.float32)
++            window_tokens = mx.array(0.0, dtype=mx.float32)
++            window_time = 0.0
++
++        if it % args.steps_per_save == 0 and rank == 0:
++            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
++            mx.save_safetensors(str(args.adapter_file), adapter_weights)
++            checkpoint = (
++                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
++            )
++            mx.save_safetensors(str(checkpoint), adapter_weights)
++            tqdm.write(
++                f"\n"
++                f"Distill iter {it}: Saved adapter weights to "
++                f"{args.adapter_file} and {checkpoint}."
++            )
++
++        tic = time.perf_counter()
++
++    if rank == 0:
++        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
++        mx.save_safetensors(str(args.adapter_file), adapter_weights)
++        tqdm.write(f"Saved final weights to {args.adapter_file}.")
diff --git a/mlx_lm_lora/train.py b/mlx_lm_lora/train.py
index 9efa33a..9ea2a86 100644
--- a/mlx_lm_lora/train.py
+++ b/mlx_lm_lora/train.py
@@ -2,9 +2,9 @@ from pathlib import Path
 import importlib.util
 import argparse
 import math
-import yaml
 import sys
 import re
+import yaml
 
 import numpy as np
 
@@ -26,7 +26,8 @@ from .trainer.rflhf_trainer import RLHFTrainingArgs, evaluate_rlhf, train_rlhf
 from .trainer.xpo_trainer import  XPOTrainingArgs, evaluate_xpo, train_xpo
 from .trainer.dpo_trainer import DPOTrainingArgs, evaluate_dpo, train_dpo
 from .trainer.cpo_trainer import CPOTrainingArgs, evaluate_cpo, train_cpo
-from .trainer.datasets import CacheDataset, load_dataset
+from .trainer.distill_trainer import DistillationTrainingArgs, train_on_policy_distill
+from .trainer.datasets import CacheDataset, load_dataset, load_prompt_only_dataset
 from .utils import fuse_and_save_model, from_pretrained
 
 from mlx_lm.tuner.utils import (
@@ -70,6 +71,7 @@ CONFIG_DEFAULTS = {
         "adafactor": {},
     },
     "data": "data/",
+    "distill_prompts_data": None,
     "seed": 0,
     "num_layers": 16,
     "batch_size": 4,
@@ -86,9 +88,11 @@ CONFIG_DEFAULTS = {
     "test": False,
     "test_batches": 500,
     "max_seq_length": 2048,
+    "max_generation_tokens": 256,
     "config": None,
     "grad_checkpoint": False,
     "lr_schedule": None,
+    "training_schedule": None,
     "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 10.0},
     "mask_prompt": False,
     "fuse": True,
@@ -106,6 +110,8 @@ CONFIG_DEFAULTS = {
     "judge": None,
     "judge_config": {},
     "alpha": 1e-5,
+    "teacher_model": None,
+    "distill_temperature": 0.0,
 
     # GRPO args
     "group_size": 4,
@@ -147,6 +153,65 @@ def calculate_iters(train_set, batch_size, epochs) -> int:
     return iters
 
 
+def parse_training_schedule(schedule_str: str):
+    if schedule_str is None:
+        return None
+    entries = []
+    for raw_part in schedule_str.split(","):
+        part = raw_part.strip()
+        if not part:
+            continue
+        if ":" not in part:
+            raise ValueError(
+                f"Invalid training schedule entry '{part}'. Expected format mode:weight."
+            )
+        mode, weight = part.split(":", 1)
+        try:
+            weight_val = float(weight.strip())
+        except ValueError as exc:
+            raise ValueError(f"Invalid weight '{weight}' in training schedule.") from exc
+        entries.append({"mode": mode.strip(), "weight": weight_val})
+    if not entries:
+        raise ValueError("Training schedule was provided but no valid entries were parsed.")
+    if any(item["weight"] <= 0 for item in entries):
+        raise ValueError("Training schedule weights must be positive.")
+    return entries
+
+
+def allocate_schedule_iterations(total_iters: int, entries):
+    if total_iters is None:
+        raise ValueError("Total iterations must be specified when using a training schedule.")
+    total_weight = sum(item["weight"] for item in entries)
+    if total_weight <= 0:
+        raise ValueError("Training schedule weights must sum to a positive value.")
+
+    exact_counts = [
+        (item, total_iters * item["weight"] / total_weight) for item in entries
+    ]
+    assigned_counts = []
+    residuals = []
+    for item, exact in exact_counts:
+        count = int(math.floor(exact))
+        assigned_counts.append([item, count])
+        residuals.append((exact - count, item))
+
+    assigned_total = sum(count for _, count in assigned_counts)
+    remaining = total_iters - assigned_total
+    residuals.sort(key=lambda tup: tup[0], reverse=True)
+
+    idx = 0
+    while remaining > 0 and residuals:
+        _, item = residuals[idx % len(residuals)]
+        for entry in assigned_counts:
+            if entry[0] is item:
+                entry[1] += 1
+                remaining -= 1
+                break
+        idx += 1
+
+    return [(entry["mode"], count) for entry, count in assigned_counts]
+
+
 def build_parser():
     parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
     parser.add_argument(
@@ -154,6 +219,12 @@ def build_parser():
         type=str,
         help="The path to the local model directory or Hugging Face repo.",
     )
+    parser.add_argument(
+        "--teacher-model",
+        type=str,
+        default=None,
+        help="Optional teacher model path for on-policy distillation.",
+    )
     parser.add_argument(
         "--load-in-4bits",
         action="store_true",
@@ -188,6 +259,12 @@ def build_parser():
             "of a Hugging Face dataset (e.g., 'mlx-community/wikisql')"
         ),
     )
+    parser.add_argument(
+        "--distill-prompts-data",
+        type=str,
+        default=None,
+        help="Directory or dataset id providing prompts for on-policy distillation.",
+    )
     parser.add_argument(
         "--train-type",
         type=str,
@@ -198,8 +275,14 @@ def build_parser():
         "--train-mode",
         type=str,
         default="sft",
-        choices=["sft", "dpo", "cpo", "orpo", "grpo", "online_dpo", "xpo", "rlhf"],
-        help="Training mode: sft, dpo, rlhf, online_dpo, xpo, cpo, orpo, or grpo, default is sft",
+        choices=["sft", "dpo", "cpo", "orpo", "grpo", "online_dpo", "xpo", "rlhf", "distill_on_policy"],
+        help="Training mode: sft, dpo, rlhf, online_dpo, xpo, cpo, orpo, grpo, or distill_on_policy. Default is sft",
+    )
+    parser.add_argument(
+        "--training-schedule",
+        type=str,
+        default=None,
+        help="Comma-separated list of mode:weight pairs to sequence training phases.",
     )
     parser.add_argument(
         "--optimizer",
@@ -295,6 +378,18 @@ def build_parser():
         type=int,
         help="Maximum sequence length.",
     )
+    parser.add_argument(
+        "--max-generation-tokens",
+        type=int,
+        default=256,
+        help="Maximum number of tokens to generate during on-policy distillation.",
+    )
+    parser.add_argument(
+        "--distill-temperature",
+        type=float,
+        default=0.0,
+        help="Sampling temperature when generating student rollouts for distillation.",
+    )
     parser.add_argument(
         "-c",
         "--config",
@@ -455,6 +550,8 @@ def train_model(
     train_set,
     valid_set,
     training_callback: TrainingCallback = None,
+    teacher_model: nn.Module = None,
+    distill_dataset=None,
 ):
     mx.random.seed(args.seed)
 
@@ -567,6 +664,71 @@ def train_model(
 
     opt = opt_class(learning_rate=lr, **optimizer_config)
 
+    schedule_entries = getattr(args, "_parsed_training_schedule", None)
+    if schedule_entries:
+        iteration_plan = allocate_schedule_iterations(args.iters, schedule_entries)
+        sft_train_cache = CacheDataset(train_set) if train_set and len(train_set) else None
+        sft_valid_cache = CacheDataset(valid_set) if valid_set and len(valid_set) else None
+        distill_cache = CacheDataset(distill_dataset) if distill_dataset and len(distill_dataset) else None
+
+        for mode, count in iteration_plan:
+            if count <= 0:
+                continue
+            if mode == "sft":
+                if sft_train_cache is None or sft_valid_cache is None:
+                    raise ValueError("SFT dataset is required but not available for the training schedule.")
+                sft_training_args = SFTTrainingArgs(
+                    batch_size=args.batch_size,
+                    iters=count,
+                    val_batches=args.val_batches,
+                    steps_per_report=args.steps_per_report,
+                    steps_per_eval=args.steps_per_eval,
+                    steps_per_save=args.save_every,
+                    adapter_file=adapter_file,
+                    max_seq_length=args.max_seq_length,
+                    grad_checkpoint=args.grad_checkpoint,
+                    gradient_accumulation_steps=args.gradient_accumulation_steps,
+                )
+                train_sft(
+                    model=model,
+                    args=sft_training_args,
+                    optimizer=opt,
+                    train_dataset=sft_train_cache,
+                    val_dataset=sft_valid_cache,
+                    training_callback=training_callback,
+                )
+            elif mode == "distill_on_policy":
+                if teacher_model is None:
+                    raise ValueError("Teacher model required for distillation in training schedule.")
+                if distill_cache is None:
+                    raise ValueError("Distillation dataset required for training schedule entry.")
+                distill_args = DistillationTrainingArgs(
+                    batch_size=args.batch_size,
+                    iters=count,
+                    val_batches=0,
+                    steps_per_report=args.steps_per_report,
+                    steps_per_eval=args.steps_per_eval,
+                    steps_per_save=args.save_every,
+                    adapter_file=adapter_file,
+                    max_seq_length=args.max_seq_length,
+                    grad_checkpoint=False,
+                    gradient_accumulation_steps=args.gradient_accumulation_steps,
+                    max_generation_tokens=args.max_generation_tokens,
+                    distill_temperature=args.distill_temperature,
+                )
+                train_on_policy_distill(
+                    model=model,
+                    teacher_model=teacher_model,
+                    tokenizer=tokenizer,
+                    optimizer=opt,
+                    train_dataset=distill_cache,
+                    args=distill_args,
+                    training_callback=training_callback,
+                )
+            else:
+                raise ValueError(f"Unsupported mode '{mode}' in training schedule.")
+        return
+
     if args.train_mode == "orpo":
         orpo_training_args = ORPOTrainingArgs(
             batch_size=args.batch_size,
@@ -864,6 +1026,35 @@ def train_model(
             training_callback=training_callback,
         )
 
+    elif args.train_mode == "distill_on_policy":
+        if teacher_model is None:
+            raise ValueError("Teacher model must be provided for distill_on_policy training.")
+        if distill_dataset is None or len(distill_dataset) == 0:
+            raise ValueError("A non-empty distillation dataset is required for distill_on_policy training.")
+        distill_args = DistillationTrainingArgs(
+            batch_size=args.batch_size,
+            iters=args.iters,
+            val_batches=0,
+            steps_per_report=args.steps_per_report,
+            steps_per_eval=args.steps_per_eval,
+            steps_per_save=args.save_every,
+            adapter_file=adapter_file,
+            max_seq_length=args.max_seq_length,
+            grad_checkpoint=False,
+            gradient_accumulation_steps=args.gradient_accumulation_steps,
+            max_generation_tokens=args.max_generation_tokens,
+            distill_temperature=args.distill_temperature,
+        )
+        train_on_policy_distill(
+            model=model,
+            teacher_model=teacher_model,
+            tokenizer=tokenizer,
+            optimizer=opt,
+            train_dataset=CacheDataset(distill_dataset),
+            args=distill_args,
+            training_callback=training_callback,
+        )
+
     elif args.train_mode == "sft":
         sft_training_args = SFTTrainingArgs(
             batch_size=args.batch_size,
@@ -935,6 +1126,9 @@ def evaluate_model(args, model: nn.Module, tokenizer, test_set):
         for metric_name, metric_value in test_metrics.items():
             print(f"  {metric_name}: {float(metric_value):.3f}")
 
+    elif args.train_mode == "distill_on_policy":
+        print("Evaluation for distill_on_policy training is not currently implemented.")
+
     elif args.train_mode == "rlhf":
         if args.reference_model_path:
             reference_model, _ = load(args.reference_model_path)
@@ -1072,21 +1266,91 @@ def run(args, training_callback: TrainingCallback = None):
     # model, tokenizer = load(args.model)
 
     if args.load_in_4bits:
-        quanziation_config = {"bits": 4, "group_size": 64}
+        quantization_config = {"bits": 4, "group_size": 64}
     elif args.load_in_6bits:
-        quanziation_config = {"bits": 6, "group_size": 64}
+        quantization_config = {"bits": 6, "group_size": 64}
     elif args.load_in_8bits:
-        quanziation_config = {"bits": 8, "group_size": 64}
+        quantization_config = {"bits": 8, "group_size": 64}
     else:
-        quanziation_config = None
+        quantization_config = None
 
     model, tokenizer = from_pretrained(
         model=args.model,
-        quantized_load=quanziation_config,
+        quantized_load=quantization_config,
     )
 
-    print("Loading datasets")
-    train_set, valid_set, test_set = load_dataset(args, tokenizer)
+    schedule_entries = None
+    raw_schedule = getattr(args, "training_schedule", None)
+    if raw_schedule:
+        if isinstance(raw_schedule, str):
+            schedule_entries = parse_training_schedule(raw_schedule)
+        elif isinstance(raw_schedule, list):
+            schedule_entries = raw_schedule
+        else:
+            raise ValueError("training_schedule must be a string or list of schedule entries.")
+    setattr(args, "_parsed_training_schedule", schedule_entries)
+
+    requires_distill = args.train_mode == "distill_on_policy"
+    if schedule_entries:
+        requires_distill = requires_distill or any(
+            entry["mode"] == "distill_on_policy" for entry in schedule_entries
+        )
+
+    teacher_model = None
+    distill_dataset = None
+    if args.train and requires_distill:
+        if args.teacher_model is None:
+            raise ValueError("A teacher model must be provided when using on-policy distillation.")
+        teacher_model, teacher_tokenizer = from_pretrained(args.teacher_model)
+
+        def _extract_vocab(tok):
+            if hasattr(tok, "get_vocab"):
+                return tok.get_vocab()
+            return getattr(tok, "vocab", None)
+
+        student_vocab = _extract_vocab(tokenizer)
+        teacher_vocab = _extract_vocab(teacher_tokenizer)
+        if student_vocab is not None and teacher_vocab is not None:
+            if student_vocab != teacher_vocab:
+                raise ValueError("Student and teacher tokenizers appear to use different token-id mappings.")
+        else:
+            special_tokens = [
+                "bos_token_id",
+                "eos_token_id",
+                "pad_token_id",
+                "unk_token_id",
+                "sep_token_id",
+            ]
+            for attr in special_tokens:
+                student_id = getattr(tokenizer, attr, None)
+                teacher_id = getattr(teacher_tokenizer, attr, None)
+                if None not in (student_id, teacher_id) and student_id != teacher_id:
+                    raise ValueError(
+                        f"Student and teacher tokenizers must agree on {attr.replace('_', ' ')}."
+                    )
+
+        if args.distill_prompts_data is None:
+            raise ValueError("distill_prompts_data must be provided for on-policy distillation.")
+        distill_dataset = load_prompt_only_dataset(args.distill_prompts_data, tokenizer, args)
+        if len(distill_dataset) == 0:
+            raise ValueError("Distillation dataset is empty.")
+
+    need_train_data = True
+    if args.train and not args.test:
+        if schedule_entries:
+            schedule_modes = {entry["mode"] for entry in schedule_entries}
+            if schedule_modes <= {"distill_on_policy"}:
+                need_train_data = False
+        elif args.train_mode == "distill_on_policy":
+            need_train_data = False
+
+    if need_train_data or args.test:
+        print("Loading datasets")
+        train_set, valid_set, test_set = load_dataset(args, tokenizer)
+    else:
+        train_set = []
+        valid_set = []
+        test_set = []
 
     if args.test and not args.train:
         if args.adapter_path != "":
@@ -1094,7 +1358,16 @@ def run(args, training_callback: TrainingCallback = None):
 
     elif args.train:
         print("Training")
-        train_model(args, model, tokenizer, train_set, valid_set, training_callback)
+        train_model(
+            args,
+            model,
+            tokenizer,
+            train_set,
+            valid_set,
+            training_callback,
+            teacher_model=teacher_model,
+            distill_dataset=distill_dataset,
+        )
     else:
         raise ValueError("Must provide at least one of --train or --test")
 
diff --git a/mlx_lm_lora/trainer/datasets.py b/mlx_lm_lora/trainer/datasets.py
index 8fb1ad4..79a6ba4 100644
--- a/mlx_lm_lora/trainer/datasets.py
+++ b/mlx_lm_lora/trainer/datasets.py
@@ -76,12 +76,28 @@ class PromptDataset:
         prompt_key: str = "prompt",
     ):
         self._data = data
-        self.chat_key = prompt_key
+        self.prompt_key = prompt_key
         self.tokenizer = tokenizer
 
     def process(self, d):
-        messages = d[self.chat_key]
-        return {"prompt": self.tokenizer.apply_chat_template(messages, add_generation_prompt=True), "prompt_text": self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)}
+        messages = d[self.prompt_key]
+        if isinstance(messages, str):
+            chat = [{"role": "user", "content": messages}]
+        elif isinstance(messages, dict):
+            chat = [messages]
+        else:
+            chat = messages
+
+        prompt_tokens = self.tokenizer.apply_chat_template(
+            chat,
+            add_generation_prompt=True,
+        )
+        prompt_text = self.tokenizer.apply_chat_template(
+            chat,
+            add_generation_prompt=True,
+            tokenize=False,
+        )
+        return {"prompt": prompt_tokens, "prompt_text": prompt_text}
 
     def __getitem__(self, idx: int):
         return self._data[idx]
@@ -452,15 +468,21 @@ def create_dataset(
                 )
         else:
             raise ValueError("Unsupported data format for Online DPO or CPO training.")
-    elif train_mode in ["online_dpo", "xpo", "rlhf"]:
+    elif train_mode in ["online_dpo", "xpo", "rlhf", "distill_on_policy"]:
         if prompt_feature in sample:
             return PromptDataset(
                 data=data,
                 tokenizer=tokenizer,
                 prompt_key=prompt_feature,
             )
+        if chat_feature in sample:
+            return PromptDataset(
+                data=data,
+                tokenizer=tokenizer,
+                prompt_key=chat_feature,
+            )
         else:
-            raise ValueError("Unsupported data format for Online DPO or XPO training.")
+            raise ValueError("Unsupported data format for Online DPO, XPO, or distill_on_policy training.")
     elif train_mode in ["grpo"]:
         if prompt_feature in sample:
             return GRPODataset(
@@ -618,3 +640,33 @@ def load_dataset(args, tokenizer: PreTrainedTokenizer):
             "Test set not found or empty. Must provide test set for evaluation."
         )
     return train, valid, test
+
+
+def load_prompt_only_dataset(
+    data_source: str,
+    tokenizer: PreTrainedTokenizer,
+    args,
+):
+    """
+    Load a prompt-only dataset for on-policy distillation. Accepts either plain
+    prompt strings (``{"prompt": ...}``) or chat-style prompts
+    (``{"messages": [...]}``).
+    """
+    config = types.SimpleNamespace(
+        train_mode="online_dpo",
+        prompt_feature=getattr(args, "prompt_feature", "prompt"),
+        chat_feature=getattr(args, "chat_feature", "messages"),
+        system_feature=getattr(args, "system_feature", "system"),
+        mask_prompt=False,
+    )
+
+    data_path = Path(data_source)
+    if data_path.exists():
+        train, _, _ = load_local_dataset(data_path, tokenizer, config)
+    else:
+        train, _, _ = load_hf_dataset(data_source, tokenizer, config)
+
+    if len(train) == 0:
+        raise ValueError("Distillation dataset is empty.")
+
+    return train
diff --git a/mlx_lm_lora/trainer/distill_trainer.py b/mlx_lm_lora/trainer/distill_trainer.py
new file mode 100644
index 0000000..87259b0
--- /dev/null
+++ b/mlx_lm_lora/trainer/distill_trainer.py
@@ -0,0 +1,376 @@
+from dataclasses import dataclass, field
+from pathlib import Path
+from typing import List, Tuple, Optional
+import time
+
+import numpy as np
+from tqdm import tqdm
+
+from mlx.utils import tree_flatten
+from mlx.nn.utils import average_gradients
+import mlx.core as mx
+import mlx.nn as nn
+
+from mlx_lm.generate import generate
+from mlx_lm.sample_utils import make_sampler
+from mlx_lm.tuner.callbacks import TrainingCallback
+
+from .datasets import CacheDataset
+from .online_dpo_trainer import iterate_online_dpo_batches
+from .sft_trainer import SFTTrainingArgs
+
+
+@dataclass
+class DistillationTrainingArgs(SFTTrainingArgs):
+    max_generation_tokens: int = field(
+        default=256,
+        metadata={"help": "Maximum number of tokens to generate per prompt."},
+    )
+    distill_temperature: float = field(
+        default=0.0,
+        metadata={"help": "Sampling temperature for student rollouts."},
+    )
+
+
+def _sample_student_responses(
+    model: nn.Module,
+    tokenizer,
+    prompt_texts: List[str],
+    prompt_token_sequences: List[List[int]],
+    max_tokens: int,
+    temperature: float,
+    max_output_tokens: List[int],
+) -> List[List[int]]:
+    eos_tokens = getattr(tokenizer, "eos_token_ids", None)
+    if eos_tokens is None:
+        single_eos = getattr(tokenizer, "eos_token_id", None)
+        eos_tokens = [single_eos] if single_eos is not None else []
+    eos_tokens = [tok for tok in eos_tokens if tok is not None]
+
+    sampler = make_sampler(
+        temperature,
+        top_p=1.0,
+        min_p=0.0,
+        min_tokens_to_keep=1,
+        top_k=0,
+        xtc_probability=0.0,
+        xtc_threshold=0.0,
+        xtc_special_tokens=tokenizer.encode("\n") + eos_tokens,
+    )
+
+    completions: List[List[int]] = []
+    for prompt_text, prompt_tokens, allowed_tokens in zip(
+        prompt_texts, prompt_token_sequences, max_output_tokens
+    ):
+        allowed_tokens = max(0, allowed_tokens)
+        if allowed_tokens == 0:
+            completions.append([])
+            continue
+        if max_tokens is None:
+            current_max = max(1, allowed_tokens)
+        else:
+            current_max = max(1, min(max_tokens, allowed_tokens))
+
+        completion = generate(
+            model=model,
+            tokenizer=tokenizer,
+            prompt=prompt_text,
+            max_tokens=current_max,
+            sampler=sampler,
+            verbose=False,
+        )
+        if isinstance(completion, str):
+            completion_ids = list(tokenizer.encode(completion))
+        elif isinstance(completion, (list, tuple)):
+            completion_ids = list(completion)
+        else:
+            completion_ids = list(completion)
+
+        prompt_encoded = tokenizer.encode(prompt_text)
+        prefix_len = len(prompt_encoded)
+        if prefix_len and len(completion_ids) >= prefix_len and completion_ids[:prefix_len] == prompt_encoded:
+            completion_ids = completion_ids[prefix_len:]
+        else:
+            prompt_length = len(prompt_tokens)
+            if (
+                prompt_length > 0
+                and len(completion_ids) >= prompt_length
+                and completion_ids[:prompt_length] == prompt_tokens
+            ):
+                completion_ids = completion_ids[prompt_length:]
+
+        if current_max is not None:
+            completion_ids = completion_ids[:current_max]
+        completions.append(list(completion_ids))
+    return completions
+
+
+def _prepare_distill_inputs(
+    model: nn.Module,
+    tokenizer,
+    batch,
+    max_tokens: int,
+    temperature: float,
+    max_seq_length: int,
+) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
+    prompts, prompt_texts = batch
+    prompt_token_sequences = []
+    for p in prompts:
+        if hasattr(p, "tolist"):
+            prompt_token_sequences.append(list(p.tolist()))
+        elif isinstance(p, (list, tuple)):
+            prompt_token_sequences.append(list(p))
+        else:
+            prompt_token_sequences.append([int(p)])
+
+    allowed_token_budget: List[int] = []
+    for prompt_tokens in prompt_token_sequences:
+        headroom = max(0, max_seq_length - len(prompt_tokens))
+        allowed_token_budget.append(headroom)
+
+    was_training = model.training
+    model.eval()
+    try:
+        completions = _sample_student_responses(
+            model=model,
+            tokenizer=tokenizer,
+            prompt_texts=prompt_texts,
+            prompt_token_sequences=prompt_token_sequences,
+            max_tokens=max_tokens,
+            temperature=temperature,
+            max_output_tokens=allowed_token_budget,
+        )
+    finally:
+        if was_training:
+            model.train()
+
+    sequences: List[List[int]] = []
+    prompt_lengths: List[int] = []
+    lengths: List[int] = []
+
+    for prompt_tokens, completion_ids in zip(prompt_token_sequences, completions):
+        allowed_total = max(0, max_seq_length - len(prompt_tokens))
+        trimmed_completion = completion_ids[:allowed_total]
+        sequence = prompt_tokens + trimmed_completion
+        if len(sequence) <= len(prompt_tokens):
+            # No generated tokens; skip this sample
+            continue
+        sequences.append(sequence)
+        prompt_lengths.append(len(prompt_tokens))
+        lengths.append(len(sequence))
+
+    if not sequences:
+        return None
+
+    max_len = max(lengths)
+    batch_size = len(sequences)
+    inputs = np.zeros((batch_size, max_len), dtype=np.int32)
+    for i, seq in enumerate(sequences):
+        inputs[i, : len(seq)] = seq
+
+    prompt_lengths = np.array(prompt_lengths, dtype=np.int32)
+    lengths = np.array(lengths, dtype=np.int32)
+    return inputs, prompt_lengths, lengths
+
+
+def _compute_generation_mask(
+    prompt_lengths: mx.array,
+    total_lengths: mx.array,
+    max_length: int,
+) -> mx.array:
+    token_positions = mx.arange(max_length - 1)[None, :]
+    prompt_offsets = mx.maximum(prompt_lengths - 1, 0)[:, None]
+    sequence_limits = mx.maximum(total_lengths - 1, 0)[:, None]
+    mask = mx.logical_and(token_positions >= prompt_offsets, token_positions < sequence_limits)
+    return mask.astype(mx.float32)
+
+
+def train_on_policy_distill(
+    model: nn.Module,
+    teacher_model: nn.Module,
+    tokenizer,
+    optimizer,
+    train_dataset: CacheDataset,
+    args: DistillationTrainingArgs,
+    training_callback: TrainingCallback = None,
+):
+    teacher_model.eval()
+    if hasattr(teacher_model, "freeze"):
+        teacher_model.freeze()
+    elif hasattr(teacher_model, "parameters"):
+        for param in teacher_model.parameters():
+            if hasattr(param, "requires_grad"):
+                param.requires_grad = False
+    model.train()
+
+    world = mx.distributed.init()
+    rank = world.rank()
+
+    if args.gradient_accumulation_steps != 1 and rank == 0:
+        tqdm.write(
+            "[distill] gradient_accumulation_steps > 1 detected; overriding to 1 for distillation phase."
+        )
+
+    iterator = iterate_online_dpo_batches(
+        dataset=train_dataset,
+        batch_size=args.batch_size,
+        max_seq_length=args.max_seq_length,
+        train=True,
+    )
+
+    def distill_loss(model, inputs, mask):
+        student_logits = model(inputs).astype(mx.float32)
+        teacher_logits = mx.stop_gradient(teacher_model(inputs).astype(mx.float32))
+
+        student_log_probs = nn.log_softmax(student_logits[:, :-1, :], axis=-1)
+        teacher_log_probs = nn.log_softmax(teacher_logits[:, :-1, :], axis=-1)
+        teacher_probs = mx.exp(teacher_log_probs)
+
+        targets = inputs[:, 1:]
+        kl_per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(axis=-1)
+        kl_sum = (kl_per_token * mask).sum()
+        num_tokens = mask.sum()
+        if num_tokens == 0:
+            raise ValueError("No generated tokens available for distillation loss.")
+        loss = kl_sum / num_tokens
+
+        teacher_token_logps = mx.take_along_axis(
+            teacher_log_probs, targets[..., None], axis=-1
+        ).squeeze(-1)
+        student_token_logps = mx.take_along_axis(
+            student_log_probs, targets[..., None], axis=-1
+        ).squeeze(-1)
+
+        teacher_sum = (teacher_token_logps * mask).sum()
+        student_sum = (student_token_logps * mask).sum()
+        return loss, (num_tokens, teacher_sum, student_sum, kl_sum)
+
+    loss_value_and_grad = nn.value_and_grad(model, distill_loss)
+
+    loss_numerator = mx.array(0.0, dtype=mx.float32)
+    total_tokens = mx.array(0.0, dtype=mx.float32)
+    kl_total = mx.array(0.0, dtype=mx.float32)
+    teacher_log_total = mx.array(0.0, dtype=mx.float32)
+    student_log_total = mx.array(0.0, dtype=mx.float32)
+
+    window_tokens = mx.array(0.0, dtype=mx.float32)
+    window_time = 0.0
+
+    pbar = tqdm(total=args.iters, desc="Distill", disable=rank != 0)
+    it = 0
+    tic = time.perf_counter()
+    consecutive_empty = 0
+    max_empty = 5
+    while it < args.iters:
+        batch = next(iterator)
+        prepared = _prepare_distill_inputs(
+            model=model,
+            tokenizer=tokenizer,
+            batch=batch,
+            max_tokens=args.max_generation_tokens,
+            temperature=args.distill_temperature,
+            max_seq_length=args.max_seq_length,
+        )
+        if prepared is None:
+            consecutive_empty += 1
+            if consecutive_empty >= max_empty:
+                raise RuntimeError(
+                    "Failed to generate non-empty student completions after multiple attempts."
+                )
+            tic = time.perf_counter()
+            continue
+        consecutive_empty = 0
+
+        inputs_np, prompt_lengths_np, lengths_np = prepared
+        inputs = mx.array(inputs_np)
+        prompt_lengths = mx.array(prompt_lengths_np)
+        lengths = mx.array(lengths_np)
+        mask = _compute_generation_mask(prompt_lengths, lengths, inputs.shape[1])
+
+        (loss_value, metrics), grad = loss_value_and_grad(model, inputs, mask)
+        grad = average_gradients(grad)
+        optimizer.update(model, grad)
+
+        num_tokens = metrics[0]
+        teacher_sum = metrics[1]
+        student_sum = metrics[2]
+        kl_sum = metrics[3]
+
+        loss_numerator += loss_value * num_tokens
+        total_tokens += num_tokens
+        kl_total += kl_sum
+        teacher_log_total += teacher_sum
+        student_log_total += student_sum
+        window_tokens += num_tokens
+
+        it += 1
+        pbar.update(1)
+        window_time += time.perf_counter() - tic
+
+        if it % args.steps_per_report == 0 or it == args.iters:
+            mx.eval(loss_numerator, total_tokens, kl_total, teacher_log_total, student_log_total, window_tokens)
+
+            total_loss = mx.distributed.all_sum(loss_numerator, stream=mx.cpu).item()
+            total_tok = mx.distributed.all_sum(total_tokens, stream=mx.cpu).item()
+            total_kl = mx.distributed.all_sum(kl_total, stream=mx.cpu).item()
+            total_teacher = mx.distributed.all_sum(teacher_log_total, stream=mx.cpu).item()
+            total_student = mx.distributed.all_sum(student_log_total, stream=mx.cpu).item()
+            window_tok = mx.distributed.all_sum(window_tokens, stream=mx.cpu).item()
+
+            avg_loss = total_loss / max(total_tok, 1.0)
+            avg_kl = total_kl / max(total_tok, 1.0)
+            avg_teacher = total_teacher / max(total_tok, 1.0)
+            avg_student = total_student / max(total_tok, 1.0)
+            avg_tokens_per_sec = window_tok / max(window_time, 1e-6)
+            lr_value = optimizer.learning_rate.item()
+
+            if rank == 0:
+                tqdm.write(
+                    f"\nDistill iter {it}: "
+                    f"loss {avg_loss:.4f}, "
+                    f"kl/token {avg_kl:.4f}, "
+                    f"teacher_logp {avg_teacher:.4f}, "
+                    f"student_logp {avg_student:.4f}, "
+                    f"lr {lr_value:.3e}, "
+                    f"window_tok/s {avg_tokens_per_sec:.3f}"
+                )
+
+            if training_callback is not None:
+                train_info = {
+                    "iteration": it,
+                    "train_loss": avg_loss,
+                    "learning_rate": lr_value,
+                    "tokens_per_second": avg_tokens_per_sec,
+                    "kl_per_token": avg_kl,
+                    "teacher_logp": avg_teacher,
+                    "student_logp": avg_student,
+                }
+                training_callback.on_train_loss_report(train_info)
+
+            loss_numerator = mx.array(0.0, dtype=mx.float32)
+            total_tokens = mx.array(0.0, dtype=mx.float32)
+            kl_total = mx.array(0.0, dtype=mx.float32)
+            teacher_log_total = mx.array(0.0, dtype=mx.float32)
+            student_log_total = mx.array(0.0, dtype=mx.float32)
+            window_tokens = mx.array(0.0, dtype=mx.float32)
+            window_time = 0.0
+
+        if it % args.steps_per_save == 0 and rank == 0:
+            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
+            mx.save_safetensors(str(args.adapter_file), adapter_weights)
+            checkpoint = (
+                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
+            )
+            mx.save_safetensors(str(checkpoint), adapter_weights)
+            tqdm.write(
+                f"\n"
+                f"Distill iter {it}: Saved adapter weights to "
+                f"{args.adapter_file} and {checkpoint}."
+            )
+
+        tic = time.perf_counter()
+
+    if rank == 0:
+        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
+        mx.save_safetensors(str(args.adapter_file), adapter_weights)
+        tqdm.write(f"Saved final weights to {args.adapter_file}.")
