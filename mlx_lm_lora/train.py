from pathlib import Path
import importlib.util
import argparse
import math
import sys
import re
import yaml

import numpy as np

import mlx.optimizers as optim
import mlx.core as mx
import mlx.nn as nn

from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.tuner.callbacks import WandBCallback
from mlx_lm.utils import load, save_config
from mlx_optimizers import QHAdam

from .trainer.grpo_reward_functions import get_reward_function, get_default_reward_functions, list_available_reward_functions
from .trainer.online_dpo_trainer import  OnlineDPOTrainingArgs, evaluate_online_dpo, train_online_dpo
from .trainer.sft_trainer import SFTTrainingArgs, TrainingCallback, evaluate_sft, train_sft
from .trainer.grpo_trainer import GRPOTrainingArgs, evaluate_grpo, train_grpo
from .trainer.orpo_trainer import ORPOTrainingArgs, evaluate_orpo, train_orpo
from .trainer.rflhf_trainer import RLHFTrainingArgs, evaluate_rlhf, train_rlhf
from .trainer.xpo_trainer import  XPOTrainingArgs, evaluate_xpo, train_xpo
from .trainer.dpo_trainer import DPOTrainingArgs, evaluate_dpo, train_dpo
from .trainer.cpo_trainer import CPOTrainingArgs, evaluate_cpo, train_cpo
from .trainer.distill_trainer import DistillationTrainingArgs, train_on_policy_distill
from .trainer.datasets import CacheDataset, load_dataset, load_prompt_only_dataset
from .utils import fuse_and_save_model, from_pretrained

from mlx_lm.tuner.utils import (
    build_schedule,
    linear_to_lora_layers,
    load_adapters,
    print_trainable_parameters,
)

yaml_loader = yaml.SafeLoader
yaml_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)

CONFIG_DEFAULTS = {
    "model": "mlx_model",
    "train": True,
    "load_in_4bits": False,
    "load_in_6bits": False,
    "load_in_8bits": False,
    "train_type": "lora",
    "train_mode": "sft",
    "optimizer": "adam",
    "optimizer_config": {
        "adam": {},
        "adamw": {},
        "muon": {},
        "qhadam": {},
        "sgd": {},
        "adafactor": {},
    },
    "data": "data/",
    "distill_prompts_data": None,
    "seed": 0,
    "num_layers": 16,
    "batch_size": 4,
    "iters": None,
    "epochs": None,
    "gradient_accumulation_steps": 1,
    "val_batches": 25,
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "resume_adapter_file": None,
    "adapter_path": "adapters",
    "save_every": 100,
    "test": False,
    "test_batches": 500,
    "max_seq_length": 2048,
    "max_generation_tokens": 256,
    "config": None,
    "grad_checkpoint": False,
    "lr_schedule": None,
    "training_schedule": None,
    "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 10.0},
    "mask_prompt": False,
    "fuse": True,

    # ORPO args
    "beta": 0.1,
    "reward_scaling": 1.0,

    # DPO args
    "dpo_cpo_loss_type": "sigmoid",
    "delta": 50.0,
    "reference_model_path": None,

    # Online DPO & XPO
    "judge": None,
    "judge_config": {},
    "alpha": 1e-5,
    "teacher_model": None,
    "distill_temperature": 0.0,

    # GRPO args
    "group_size": 4,
    "epsilon": 1e-4,
    "epsilon_high": None, # DAPO
    "max_completion_length": 512,
    "temperature": 0.8,
    "reward_weights": None,
    "reward_functions": None,
    "reward_functions_file": None,
    "grpo_loss_type": "grpo",
    "importance_sampling_level": None, # GSPO
}


def load_reward_functions_from_file(file_path):
    """Load reward functions from a Python file"""
    if not file_path or not Path(file_path).exists():
        return None
    
    try:
        print(f"Loading custom reward functions from {file_path}")
        spec = importlib.util.spec_from_file_location("custom_rewards", file_path)
        custom_rewards = importlib.util.module_from_spec(spec)
        sys.modules["custom_rewards"] = custom_rewards
        spec.loader.exec_module(custom_rewards)
        print("Successfully loaded custom reward functions")
        return True
    except Exception as e:
        print(f"Error loading custom reward functions: {e}")
        return None


def calculate_iters(train_set, batch_size, epochs) -> int:
    num_samples = len(train_set)
    batches_per_epoch = math.ceil(num_samples / batch_size)
    iters = epochs * batches_per_epoch
    print(f"[INFO] Calculated {iters} iterations from {epochs} epochs (dataset size: {num_samples}, batch size: {batch_size})")
    return iters


def parse_training_schedule(schedule_str: str):
    if schedule_str is None:
        return None
    entries = []
    for raw_part in schedule_str.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"Invalid training schedule entry '{part}'. Expected format mode:weight."
            )
        mode, weight = part.split(":", 1)
        try:
            weight_val = float(weight.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid weight '{weight}' in training schedule.") from exc
        entries.append({"mode": mode.strip(), "weight": weight_val})
    if not entries:
        raise ValueError("Training schedule was provided but no valid entries were parsed.")
    if any(item["weight"] <= 0 for item in entries):
        raise ValueError("Training schedule weights must be positive.")
    return entries


def allocate_schedule_iterations(total_iters: int, entries):
    if total_iters is None:
        raise ValueError("Total iterations must be specified when using a training schedule.")
    total_weight = sum(item["weight"] for item in entries)
    if total_weight <= 0:
        raise ValueError("Training schedule weights must sum to a positive value.")

    exact_counts = [
        (item, total_iters * item["weight"] / total_weight) for item in entries
    ]
    assigned_counts = []
    residuals = []
    for item, exact in exact_counts:
        count = int(math.floor(exact))
        assigned_counts.append([item, count])
        residuals.append((exact - count, item))

    assigned_total = sum(count for _, count in assigned_counts)
    remaining = total_iters - assigned_total
    residuals.sort(key=lambda tup: tup[0], reverse=True)

    idx = 0
    while remaining > 0 and residuals:
        _, item = residuals[idx % len(residuals)]
        for entry in assigned_counts:
            if entry[0] is item:
                entry[1] += 1
                remaining -= 1
                break
        idx += 1

    return [(entry["mode"], count) for entry, count in assigned_counts]


def build_parser():
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help="Optional teacher model path for on-policy distillation.",
    )
    parser.add_argument(
        "--load-in-4bits",
        action="store_true",
        help="Load the model in 4-bit quantization.",
        default=None,
    )
    parser.add_argument(
        "--load-in-6bits",
        action="store_true",
        help="Load the model in 6-bit quantization.",
        default=None,
    )
    parser.add_argument(
        "--load-in-8bits",
        action="store_true",
        help="Load the model in 8-bit quantization.",
        default=None,
    )

    # Training args
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
        default=None,
    )
    parser.add_argument(
        "--data",
        type=str,
        help=(
            "Directory with {train, valid, test}.jsonl files or the name "
            "of a Hugging Face dataset (e.g., 'mlx-community/wikisql')"
        ),
    )
    parser.add_argument(
        "--distill-prompts-data",
        type=str,
        default=None,
        help="Directory or dataset id providing prompts for on-policy distillation.",
    )
    parser.add_argument(
        "--train-type",
        type=str,
        choices=["lora", "dora", "full"],
        help="Type of fine-tuning to perform: lora, dora, or full.",
    )
    parser.add_argument(
        "--train-mode",
        type=str,
        default="sft",
        choices=["sft", "dpo", "cpo", "orpo", "grpo", "online_dpo", "xpo", "rlhf", "distill_on_policy"],
        help="Training mode: sft, dpo, rlhf, online_dpo, xpo, cpo, orpo, grpo, or distill_on_policy. Default is sft",
    )
    parser.add_argument(
        "--training-schedule",
        type=str,
        default=None,
        help="Comma-separated list of mode:weight pairs to sequence training phases.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw", "qhadam", "muon", "sgd", "adafactor"],
        default=None,
        help="Optimizer to use for training",
    )
    parser.add_argument(
        "--sgd-momentum",
        type=float,
        default=None,
        help="Momentum factor for SGD (requires --optimizer sgd)",
    )
    parser.add_argument(
        "--sgd-nesterov",
        action="store_true",
        default=None,
        help="Enable Nesterov momentum for SGD",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay to apply when supported by the optimizer",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        choices=["cosine", "constant"],
        default=None,
        help="Learning rate schedule to apply",
    )
    parser.add_argument(
        "--mask-prompt",
        action="store_true",
        help="Mask the prompt in the loss when training",
        default=None,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of layers to fine-tune. Default is 16, use -1 for all.",
    )
    parser.add_argument("--batch-size", type=int, help="Minibatch size.")
    parser.add_argument("--iters", type=int, help="Iterations to train for.")
    parser.add_argument("--epochs", type=int, help="Epochs to train for. Ignored if --iters is provided.")
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Number of gradient accumulation steps.", default=1)
    parser.add_argument(
        "--val-batches",
        type=int,
        help="Number of validation batches, -1 uses the entire validation set."
    )
    parser.add_argument("--learning-rate", type=float, help="Adam learning rate.")
    parser.add_argument(
        "--steps-per-report",
        type=int,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        type=str,
        help="Load path to resume training from the given fine-tuned weights.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Save/load path for the fine-tuned weights.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        help="Save the model every N iterations.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
        default=None,
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--max-generation-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate during on-policy distillation.",
    )
    parser.add_argument(
        "--distill-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature when generating student rollouts for distillation.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="A YAML configuration file with the training options",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Use gradient checkpointing to reduce memory use.",
        default=None,
    )
    parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="WandB project name to report training metrics. Disabled if None.",
    )
    parser.add_argument("--seed", type=int, help="The PRNG seed")
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="Fuse and save the trained model.",
        default=None,
    )

    # ORPO args
    parser.add_argument(
        "--beta",
        type=float,
        help="Temperature parameter for ORPO training.",
        default=0.1,
    )
    parser.add_argument(
        "--reward-scaling",
        type=float,
        help="Reward scaling factor for ORPO training, not implemented.",
        default=1.0,
    )

    # DPO args
    parser.add_argument(
        "--dpo-cpo-loss-type",
        type=str,
        help="DPO loss type: 'sigmoid', 'hinge', 'ipo', or 'dpop'.",
        choices=["sigmoid", "hinge", "ipo", "dpop"],
        default="sigmoid",
    )
    parser.add_argument(
        "--delta", type=float, help="Delta parameter for DPOP loss type.", default=50.0
    )
    parser.add_argument(
        "--reference-model-path",
        type=str,
        help="Path to reference model weights. If None, uses the same model.",
        default=None,
    )

    # Online DPO & XPO args
    parser.add_argument(
        "--judge", type=str, help="Judge to use can be a model ID or 'human'.", default="mlx-community/Josiefied-Qwen2.5-7B-Instruct-abliterated-v2-4-bit"
    )
    parser.add_argument(
        "--alpha", type=list[float], help="Judge to use can be a model ID or 'human'.", default=[1e-5]
    )

    # GRPO args
    parser.add_argument(
        "--group-size",
        type=int,
        help="Number of generations.",
        default=4,
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        help="Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.",
        default=512,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="The Epsilon for numerical stability.",
        default=1e-4,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for sampling. The higher the temperature, the more random the completions.",
        default=1.0,
    )
    parser.add_argument(
        "--reward-weights",
        type=str,
        help="Weights for each reward function. Must match the number of reward functions and be in this format [0.1, 0.2, 0.3, 0.4, 0.5]. If not given, all rewards are weighted equally with weight `1.0`.",
        default=None,
    )
    parser.add_argument(
        "--reward-functions",
        type=str,
        help=(
            "Comma-separated list of reward function names to use. These must be registered in the reward_functions registry. "
            "Use --list-reward-functions to see available functions. "
            "Example: r1_accuracy_reward_func,action_format_reward_func"
        ),
        default=None,
    )
    parser.add_argument(
        "--reward-functions-file",
        type=str,
        help=(
            "Path to a Python file containing custom reward functions. "
            "The file should define functions decorated with @register_reward_function(). "
            "Example: path/to/my_reward_functions.py"
        ),
        default=None,
    )
    parser.add_argument(
        "--list-reward-functions",
        action="store_true",
        help="List all available reward functions and exit",
    )

    parser.add_argument(
        "--grpo-loss-type",
        type=str,
        help="GRPO loss type: 'grpo', 'bnpo', or 'dr_grpo'.",
        choices=["grpo", "bnpo", "dr_grpo"],
        default="grpo",
    )

    # DAPO args
    parser.add_argument(
        "--epsilon-high",
        type=float,
        help="Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound specified in argument epsilon.",
        default=None,
    )

    # GSPO args
    parser.add_argument(
        "--importance-sampling-level",
        type=str,
        choices=["token", "sequence", None],
        default=None,
        help=(
            "Level of importance sampling to use. "
            "'token' uses token-level importance sampling, 'sequence' uses sequence-level, and None (default) disables it."
        ),
    )
    return parser


def train_model(
    args,
    model: nn.Module,
    tokenizer,
    train_set,
    valid_set,
    training_callback: TrainingCallback = None,
    teacher_model: nn.Module = None,
    distill_dataset=None,
):
    mx.random.seed(args.seed)

    if args.iters is None and args.epochs is not None:
        args.iters = calculate_iters(train_set=train_set, batch_size=args.batch_size, epochs=args.epochs)

    model.freeze()
    if args.num_layers > len(model.layers):
        raise ValueError(
            f"Requested to train {args.num_layers} layers "
            f"but the model only has {len(model.layers)} layers."
        )

    if args.train_type == "full":
        for l in model.layers[-max(args.num_layers, 0) :]:
            l.unfreeze()
    elif args.train_type in ["lora", "dora"]:
        # Convert linear layers to lora/dora layers and unfreeze in the process
        linear_to_lora_layers(
            model,
            args.num_layers,
            args.lora_parameters,
            use_dora=(args.train_type == "dora"),
        )
    else:
        raise ValueError(f"Received unknown train-type {args.train_type}")

    # Resume from weights if provided
    if args.resume_adapter_file is not None:
        print(f"Loading fine-tuned weights from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    print_trainable_parameters(model)

    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)

    adapter_file = adapter_path / "adapters.safetensors"
    save_config(vars(args), adapter_path / "adapter_config.json")

    # Initialize the selected optimizer
    lr_schedule_config = args.lr_schedule
    schedule_spec = lr_schedule_config

    if isinstance(lr_schedule_config, str):
        schedule_name = lr_schedule_config.lower()
        schedule_spec = schedule_name
        if schedule_name == "constant":
            lr_schedule_config = None
        elif schedule_name == "cosine":
            if args.iters is None:
                raise ValueError(
                    "Cosine learning rate schedule requires the total number of iterations; "
                    "specify --iters or --epochs."
                )
            lr_schedule_config = {
                "name": "cosine_decay",
                "arguments": [args.learning_rate, args.iters],
            }
        else:
            raise ValueError(f"Unsupported learning rate schedule: {schedule_name}")

    if lr_schedule_config:
        lr = build_schedule(lr_schedule_config)
        args.lr_schedule = lr_schedule_config
    else:
        lr = args.learning_rate
        args.lr_schedule = schedule_spec if schedule_spec == "constant" else None

    optimizer_name = args.optimizer.lower()
    optimizer_config = dict(args.optimizer_config.get(optimizer_name, {}))

    weight_decay_supported = {"adamw", "muon", "sgd"}
    if optimizer_name in weight_decay_supported:
        if args.weight_decay is not None:
            optimizer_config["weight_decay"] = args.weight_decay
        else:
            optimizer_config.setdefault("weight_decay", 0.0)
        args.weight_decay = optimizer_config.get("weight_decay")

    if optimizer_name == "adam":
        opt_class = optim.Adam
    elif optimizer_name == "adamw":
        opt_class = optim.AdamW
    elif optimizer_name == "qhadam":
        opt_class = QHAdam
    elif optimizer_name == "muon":
        opt_class = optim.Muon
    elif optimizer_name == "sgd":
        opt_class = optim.SGD
        if args.sgd_momentum is not None:
            optimizer_config["momentum"] = args.sgd_momentum
        elif "momentum" not in optimizer_config:
            optimizer_config["momentum"] = 0.0

        if args.sgd_nesterov is not None:
            optimizer_config["nesterov"] = args.sgd_nesterov
        elif "nesterov" not in optimizer_config:
            optimizer_config["nesterov"] = False
    elif optimizer_name == "adafactor":
        opt_class = optim.Adafactor
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    if optimizer_name == "sgd":
        if args.sgd_momentum is None:
            args.sgd_momentum = optimizer_config.get("momentum", 0.0)
        if args.sgd_nesterov is None:
            args.sgd_nesterov = optimizer_config.get("nesterov", False)

    opt = opt_class(learning_rate=lr, **optimizer_config)

    schedule_entries = getattr(args, "_parsed_training_schedule", None)
    if schedule_entries:
        iteration_plan = allocate_schedule_iterations(args.iters, schedule_entries)
        sft_train_cache = CacheDataset(train_set) if train_set and len(train_set) else None
        sft_valid_cache = CacheDataset(valid_set) if valid_set and len(valid_set) else None
        distill_cache = CacheDataset(distill_dataset) if distill_dataset and len(distill_dataset) else None

        for mode, count in iteration_plan:
            if count <= 0:
                continue
            if mode == "sft":
                if sft_train_cache is None or sft_valid_cache is None:
                    raise ValueError("SFT dataset is required but not available for the training schedule.")
                sft_training_args = SFTTrainingArgs(
                    batch_size=args.batch_size,
                    iters=count,
                    val_batches=args.val_batches,
                    steps_per_report=args.steps_per_report,
                    steps_per_eval=args.steps_per_eval,
                    steps_per_save=args.save_every,
                    adapter_file=adapter_file,
                    max_seq_length=args.max_seq_length,
                    grad_checkpoint=args.grad_checkpoint,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                )
                train_sft(
                    model=model,
                    args=sft_training_args,
                    optimizer=opt,
                    train_dataset=sft_train_cache,
                    val_dataset=sft_valid_cache,
                    training_callback=training_callback,
                )
            elif mode == "distill_on_policy":
                if teacher_model is None:
                    raise ValueError("Teacher model required for distillation in training schedule.")
                if distill_cache is None:
                    raise ValueError("Distillation dataset required for training schedule entry.")
                distill_args = DistillationTrainingArgs(
                    batch_size=args.batch_size,
                    iters=count,
                    val_batches=0,
                    steps_per_report=args.steps_per_report,
                    steps_per_eval=args.steps_per_eval,
                    steps_per_save=args.save_every,
                    adapter_file=adapter_file,
                    max_seq_length=args.max_seq_length,
                    grad_checkpoint=False,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    max_generation_tokens=args.max_generation_tokens,
                    distill_temperature=args.distill_temperature,
                )
                train_on_policy_distill(
                    model=model,
                    teacher_model=teacher_model,
                    tokenizer=tokenizer,
                    optimizer=opt,
                    train_dataset=distill_cache,
                    args=distill_args,
                    training_callback=training_callback,
                )
            else:
                raise ValueError(f"Unsupported mode '{mode}' in training schedule.")
        return

    if args.train_mode == "orpo":
        orpo_training_args = ORPOTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
            beta=args.beta,
            reward_scaling=args.reward_scaling,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        train_orpo(
            model=model,
            optimizer=opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
            args=orpo_training_args,
            training_callback=training_callback,
        )
    elif args.train_mode == "dpo":
        dpo_training_args = DPOTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
            beta=args.beta,
            loss_type=args.dpo_cpo_loss_type,
            delta=args.delta,
            reference_model_path=args.reference_model_path,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        print("Loading pretrained reference model")
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model, _ = load(args.model)

        train_dpo(
            model=model,
            ref_model=reference_model.freeze(),
            optimizer=opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
            args=dpo_training_args,
            training_callback=training_callback,
        )
    
    elif args.train_mode == "online_dpo":
        online_dpo_training_args = OnlineDPOTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
            beta=args.beta,
            loss_type=args.dpo_cpo_loss_type,
            delta=args.delta,
            reference_model_path=args.reference_model_path,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            judge=args.judge,
            max_completion_length=args.max_completion_length,
            temperature=args.temperature,
        )

        print("Loading pretrained reference model")
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model, _ = load(args.model)

        print("Loading pretrained judge model")
        if args.judge:
            if args.judge == args.reference_model_path:
                judge_model = reference_model
                judge_tokenizer = load_tokenizer(args.judge)
            else:
                judge_model, judge_tokenizer = load(args.judge)
        else:
            judge_model, judge_tokenizer = load(args.judge)

        train_online_dpo(
            model=model,
            tokenizer=tokenizer,
            ref_model=reference_model.freeze(),
            judge_model=judge_model.freeze(),
            judge_tokenizer=judge_tokenizer,
            judge_config=args.judge_config,
            optimizer=opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
            args=online_dpo_training_args,
            training_callback=training_callback,
        )
    
    elif args.train_mode == "rlhf":
        online_dpo_training_args = RLHFTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
            beta=args.beta,
            reference_model_path=args.reference_model_path,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            judge=args.judge,
            max_completion_length=args.max_completion_length,
        )

        print("Loading pretrained reference model")
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model, _ = load(args.model)

        print("Loading pretrained judge model")
        if args.judge:
            if args.judge == args.reference_model_path:
                judge_model = reference_model
                judge_tokenizer = load_tokenizer(args.judge)
            else:
                judge_model, judge_tokenizer = load(args.judge)
        else:
            judge_model, judge_tokenizer = load(args.judge)

        train_rlhf(
            model=model,
            tokenizer=tokenizer,
            ref_model=reference_model.freeze(),
            judge_model=judge_model.freeze(),
            judge_tokenizer=judge_tokenizer,
            judge_config=args.judge_config,
            optimizer=opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
            args=online_dpo_training_args,
            training_callback=training_callback,
        )

    elif args.train_mode == "xpo":
        xpo_training_args = XPOTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
            beta=args.beta,
            loss_type=args.dpo_cpo_loss_type,
            delta=args.delta,
            reference_model_path=args.reference_model_path,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            alpha=args.alpha,
            judge=args.judge,
            max_completion_length=args.max_completion_length,
        )

        print("Loading pretrained reference model")
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model, _ = load(args.model)

        print("Loading pretrained judge model")
        if args.judge:
            if args.judge == args.reference_model_path:
                judge_model = reference_model
                judge_tokenizer = load_tokenizer(args.judge)
            else:
                judge_model, judge_tokenizer = load(args.judge)
        else:
            judge_model, judge_tokenizer = load(args.judge)

        train_xpo(
            model=model,
            tokenizer=tokenizer,
            ref_model=reference_model.freeze(),
            judge_config=args.judge_config,
            judge_model=judge_model.freeze(),
            judge_tokenizer=judge_tokenizer,
            optimizer=opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
            args=xpo_training_args,
            training_callback=training_callback,
        )
    
    elif args.train_mode == "cpo":
        cpo_training_args = CPOTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
            beta=args.beta,
            loss_type=args.dpo_cpo_loss_type,
            delta=args.delta,
            reference_model_path=args.reference_model_path,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        train_cpo(
            model=model,
            optimizer=opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
            args=cpo_training_args,
            training_callback=training_callback,
        )

    elif args.train_mode == "grpo":
        if args.reward_functions_file:
            load_reward_functions_from_file(args.reward_functions_file)
        
        reward_funcs = get_default_reward_functions()
        if args.reward_functions:
            func_names = [name.strip() for name in args.reward_functions.split(',')]
            try:
                reward_funcs = [get_reward_function(name) for name in func_names]
                print(f"Using custom reward functions: {', '.join(func_names)}")
            except KeyError as e:
                print(f"Error: {str(e)}")
                print(f"Available reward functions: {list_available_reward_functions()}")
                return
            
        grpo_training_args = GRPOTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            max_completion_length=args.max_completion_length,
            grad_checkpoint=args.grad_checkpoint,
            beta=args.beta,
            group_size=args.group_size,
            epsilon=args.epsilon,
            epsilon_high=args.epsilon_high,
            reference_model_path=args.reference_model_path,
            temperature=args.temperature,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            reward_weights=(
                [float(x) for x in args.reward_weights.strip("[]").split(",")]
                if args.reward_weights
                else None
            ),
            importance_sampling_level=args.importance_sampling_level,
            grpo_loss_type=args.grpo_loss_type,
        )

        print("Loading pretrained reference model")
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        elif args.beta == 0:
            reference_model = None
        else:
            reference_model, _ = load(args.model)

        train_grpo(
            model=model,
            ref_model=reference_model.freeze() if reference_model else None,
            tokenizer=tokenizer,
            optimizer=opt,
            train_dataset=train_set,
            val_dataset=valid_set,
            reward_funcs=reward_funcs,
            args=grpo_training_args,
            training_callback=training_callback,
        )

    elif args.train_mode == "distill_on_policy":
        if teacher_model is None:
            raise ValueError("Teacher model must be provided for distill_on_policy training.")
        if distill_dataset is None or len(distill_dataset) == 0:
            raise ValueError("A non-empty distillation dataset is required for distill_on_policy training.")
        distill_args = DistillationTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=0,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=False,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_generation_tokens=args.max_generation_tokens,
            distill_temperature=args.distill_temperature,
        )
        train_on_policy_distill(
            model=model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            optimizer=opt,
            train_dataset=CacheDataset(distill_dataset),
            args=distill_args,
            training_callback=training_callback,
        )

    elif args.train_mode == "sft":
        sft_training_args = SFTTrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        train_sft(
            model=model,
            args=sft_training_args,
            optimizer=opt,
            train_dataset=CacheDataset(train_set),
            val_dataset=CacheDataset(valid_set),
            training_callback=training_callback,
        )

    else:
        raise(f"The train mode {args.train_mode} does not exist.")


def evaluate_model(args, model: nn.Module, tokenizer, test_set):
    if args.train_mode == "orpo":
        test_loss, test_rewards, _, test_metrics = evaluate_orpo(
            model=model,
            dataset=test_set,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
            beta=args.beta,
        )
        test_ppl = math.exp(test_loss)
        print(
            f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}, Rewards: {test_rewards[0]:.3f}, {test_rewards[1]:.3f}"
        )

        print("ORPO Test Metrics:")
        for metric_name, metric_value in test_metrics.items():
            print(f"  {metric_name}: {float(metric_value):.3f}")

    elif args.train_mode == "dpo":
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model = model

        test_loss, _, _, test_metrics = evaluate_dpo(
            model=model,
            ref_model=reference_model.freeze(),
            dataset=test_set,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
            beta=args.beta,
            delta=args.delta,
            loss_type=args.dpo_cpo_loss_type,
        )

        test_ppl = math.exp(test_loss)

        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}")
        print("DPO Test Metrics:")
        for metric_name, metric_value in test_metrics.items():
            print(f"  {metric_name}: {float(metric_value):.3f}")

    elif args.train_mode == "distill_on_policy":
        print("Evaluation for distill_on_policy training is not currently implemented.")

    elif args.train_mode == "rlhf":
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model, _ = load(args.model)

        test_loss, _, _, test_metrics = evaluate_rlhf(
            model=model,
            ref_model=reference_model.freeze(),
            dataset=test_set,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
            beta=args.beta,
            loss_type=args.dpo_cpo_loss_type,
            judge=args.judge,
            max_tokens=args.max_completion_length,
        )
    
    elif args.train_mode == "online_dpo":
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model, _ = load(args.model)

        test_loss, _, _, test_metrics = evaluate_online_dpo(
            model=model,
            ref_model=reference_model.freeze(),
            dataset=test_set,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
            beta=args.beta,
            delta=args.delta,
            loss_type=args.dpo_cpo_loss_type,
            judge=args.judge,
            max_tokens=args.max_completion_length,
        )
    
    elif args.train_mode == "xpo":
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model, _ = load(args.model)

        test_loss, _, _, test_metrics = evaluate_xpo(
            model=model,
            ref_model=reference_model.freeze(),
            dataset=test_set,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
            beta=args.beta,
            delta=args.delta,
            loss_type=args.dpo_cpo_loss_type,
            judge=args.judge,
            max_tokens=args.max_completion_length,
            alpha=args.alpha,
        )
    
    elif args.train_mode == "cpo":
        test_loss, _, _, test_metrics = evaluate_cpo(
            model=model,
            dataset=test_set,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
            beta=args.beta,
            delta=args.delta,
            loss_type=args.dpo_cpo_loss_type,
        )

        test_ppl = math.exp(test_loss)

        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}")
        print("CPO Test Metrics:")
        for metric_name, metric_value in test_metrics.items():
            print(f"  {metric_name}: {float(metric_value):.3f}")

    elif args.train_mode == "grpo":
        if args.reference_model_path:
            reference_model, _ = load(args.reference_model_path)
        else:
            reference_model = model

        test_loss, _, test_rewards = evaluate_grpo(
            model=model,
            ref_model=reference_model.freeze(),
            dataset=test_set,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
            beta=args.beta,
            group_size=args.group_size,
            epsilon=args.epsilon,
            epsilon_high=args.epsilon_high,
            temperature=args.temperature,
            max_tokens=args.max_seq_length,
        )

        test_ppl = math.exp(test_loss)

        rewards_str = ", ".join([f"{k}: {v:.3f}" for k, v in test_rewards.items()])
        print(
            f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}, Rewards: {rewards_str}"
        )

    elif args.train_mode == "normal":
        test_loss = evaluate_sft(
            model=model,
            dataset=CacheDataset(test_set),
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
        )

        test_ppl = math.exp(test_loss)

        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")


def run(args, training_callback: TrainingCallback = None):
    np.random.seed(args.seed)

    if args.wandb is not None:
        training_callback = WandBCallback(
            project_name=args.wandb,
            log_dir=args.adapter_path,
            config=vars(args),
            wrapped_callback=training_callback,
        )

    # print("Loading pretrained model")
    # model, tokenizer = load(args.model)

    if args.load_in_4bits:
        quantization_config = {"bits": 4, "group_size": 64}
    elif args.load_in_6bits:
        quantization_config = {"bits": 6, "group_size": 64}
    elif args.load_in_8bits:
        quantization_config = {"bits": 8, "group_size": 64}
    else:
        quantization_config = None

    model, tokenizer = from_pretrained(
        model=args.model,
        quantized_load=quantization_config,
    )

    schedule_entries = None
    raw_schedule = getattr(args, "training_schedule", None)
    if raw_schedule:
        if isinstance(raw_schedule, str):
            schedule_entries = parse_training_schedule(raw_schedule)
        elif isinstance(raw_schedule, list):
            schedule_entries = raw_schedule
        else:
            raise ValueError("training_schedule must be a string or list of schedule entries.")
    setattr(args, "_parsed_training_schedule", schedule_entries)

    requires_distill = args.train_mode == "distill_on_policy"
    if schedule_entries:
        requires_distill = requires_distill or any(
            entry["mode"] == "distill_on_policy" for entry in schedule_entries
        )

    teacher_model = None
    distill_dataset = None
    if args.train and requires_distill:
        if args.teacher_model is None:
            raise ValueError("A teacher model must be provided when using on-policy distillation.")
        teacher_model, teacher_tokenizer = from_pretrained(args.teacher_model)

        def _extract_vocab(tok):
            if hasattr(tok, "get_vocab"):
                return tok.get_vocab()
            return getattr(tok, "vocab", None)

        student_vocab = _extract_vocab(tokenizer)
        teacher_vocab = _extract_vocab(teacher_tokenizer)
        if student_vocab is not None and teacher_vocab is not None:
            if student_vocab != teacher_vocab:
                raise ValueError("Student and teacher tokenizers appear to use different token-id mappings.")
        else:
            special_tokens = [
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "unk_token_id",
                "sep_token_id",
            ]
            for attr in special_tokens:
                student_id = getattr(tokenizer, attr, None)
                teacher_id = getattr(teacher_tokenizer, attr, None)
                if None not in (student_id, teacher_id) and student_id != teacher_id:
                    raise ValueError(
                        f"Student and teacher tokenizers must agree on {attr.replace('_', ' ')}."
                    )

        if args.distill_prompts_data is None:
            raise ValueError("distill_prompts_data must be provided for on-policy distillation.")
        distill_dataset = load_prompt_only_dataset(args.distill_prompts_data, tokenizer, args)
        if len(distill_dataset) == 0:
            raise ValueError("Distillation dataset is empty.")

    need_train_data = True
    if args.train and not args.test:
        if schedule_entries:
            schedule_modes = {entry["mode"] for entry in schedule_entries}
            if schedule_modes <= {"distill_on_policy"}:
                need_train_data = False
        elif args.train_mode == "distill_on_policy":
            need_train_data = False

    if need_train_data or args.test:
        print("Loading datasets")
        train_set, valid_set, test_set = load_dataset(args, tokenizer)
    else:
        train_set = []
        valid_set = []
        test_set = []

    if args.test and not args.train:
        if args.adapter_path != "":
            load_adapters(model, args.adapter_path)

    elif args.train:
        print("Training")
        train_model(
            args,
            model,
            tokenizer,
            train_set,
            valid_set,
            training_callback,
            teacher_model=teacher_model,
            distill_dataset=distill_dataset,
        )
    else:
        raise ValueError("Must provide at least one of --train or --test")

    if args.test:
        print("Testing")
        evaluate_model(args, model, tokenizer, test_set)

    if args.fuse:
        print("Fusing model")
        fuse_and_save_model(
            model=model,
            tokenizer=tokenizer,
            save_path=args.adapter_path,
            adapter_path=None,
            de_quantize=False,
            export_gguf=False,
        )


def main(args=None):
    import os, types, yaml
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if args is None:
        parser = build_parser()
        args = parser.parse_args()
    elif isinstance(args, dict):
        # Allow programmatic overrides from notebook
        default_args = vars(build_parser().parse_args([]))
        default_args.update(args)
        args = types.SimpleNamespace(**default_args)

    if args.config:
        with open(args.config, "r") as f:
            config_args = yaml.load(f, Loader=yaml_loader)
            for k, v in config_args.items():
                if getattr(args, k, None) is None:
                    setattr(args, k, v)

    # Set all None args to defaults
    for k, v in CONFIG_DEFAULTS.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    run(args)


if __name__ == "__main__":
    main()
