from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
import time

import numpy as np
from tqdm import tqdm

from mlx.utils import tree_flatten
from mlx.nn.utils import average_gradients
import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.callbacks import TrainingCallback

from .datasets import CacheDataset
from .online_dpo_trainer import iterate_online_dpo_batches
from .sft_trainer import SFTTrainingArgs


@dataclass
class DistillationTrainingArgs(SFTTrainingArgs):
    max_generation_tokens: int = field(
        default=256,
        metadata={"help": "Maximum number of tokens to generate per prompt."},
    )
    distill_temperature: float = field(
        default=0.0,
        metadata={"help": "Sampling temperature for student rollouts."},
    )


def _sample_student_responses(
    model: nn.Module,
    tokenizer,
    prompt_texts: List[str],
    prompt_token_sequences: List[List[int]],
    max_tokens: int,
    temperature: float,
) -> List[List[int]]:
    eos_tokens = getattr(tokenizer, "eos_token_ids", None)
    if eos_tokens is None:
        single_eos = getattr(tokenizer, "eos_token_id", None)
        eos_tokens = [single_eos] if single_eos is not None else []
    eos_tokens = [tok for tok in eos_tokens if tok is not None]

    sampler = make_sampler(
        temperature,
        top_p=1.0,
        min_p=0.0,
        min_tokens_to_keep=1,
        top_k=0,
        xtc_probability=0.0,
        xtc_threshold=0.0,
        xtc_special_tokens=tokenizer.encode("\n") + eos_tokens,
    )

    completions: List[List[int]] = []
    for prompt_text, prompt_tokens in zip(prompt_texts, prompt_token_sequences):
        completion = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        if isinstance(completion, str):
            completion_ids = list(tokenizer.encode(completion))
        elif isinstance(completion, (list, tuple)):
            completion_ids = list(completion)
        else:
            completion_ids = list(completion)

        prompt_length = len(prompt_tokens)
        if (
            prompt_length > 0
            and len(completion_ids) >= prompt_length
            and completion_ids[:prompt_length] == prompt_tokens
        ):
            completion_ids = completion_ids[prompt_length:]

        if max_tokens is not None:
            completion_ids = completion_ids[:max_tokens]
        completions.append(list(completion_ids))
    return completions


def _prepare_distill_inputs(
    model: nn.Module,
    tokenizer,
    batch,
    max_tokens: int,
    temperature: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    prompts, prompt_texts = batch
    prompt_token_sequences = []
    for p in prompts:
        if hasattr(p, "tolist"):
            prompt_token_sequences.append(list(p.tolist()))
        elif isinstance(p, (list, tuple)):
            prompt_token_sequences.append(list(p))
        else:
            prompt_token_sequences.append([int(p)])

    was_training = model.training
    model.eval()
    try:
        completions = _sample_student_responses(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            prompt_token_sequences=prompt_token_sequences,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    finally:
        if was_training:
            model.train()

    sequences: List[List[int]] = []
    prompt_lengths: List[int] = []
    lengths: List[int] = []

    for prompt_tokens, completion_ids in zip(prompt_token_sequences, completions):
        sequence = prompt_tokens + completion_ids
        if len(sequence) <= len(prompt_tokens):
            # No generated tokens; skip this sample
            continue
        sequences.append(sequence)
        prompt_lengths.append(len(prompt_tokens))
        lengths.append(len(sequence))

    if not sequences:
        return None

    max_len = max(lengths)
    batch_size = len(sequences)
    inputs = np.zeros((batch_size, max_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        inputs[i, : len(seq)] = seq

    prompt_lengths = np.array(prompt_lengths, dtype=np.int32)
    lengths = np.array(lengths, dtype=np.int32)
    return inputs, prompt_lengths, lengths


def _compute_generation_mask(
    prompt_lengths: mx.array,
    total_lengths: mx.array,
    max_length: int,
) -> mx.array:
    token_positions = mx.arange(max_length - 1)[None, :]
    prompt_offsets = mx.maximum(prompt_lengths - 1, 0)[:, None]
    sequence_limits = mx.maximum(total_lengths - 1, 0)[:, None]
    mask = mx.logical_and(token_positions >= prompt_offsets, token_positions < sequence_limits)
    return mask.astype(mx.float32)


def train_on_policy_distill(
    model: nn.Module,
    teacher_model: nn.Module,
    tokenizer,
    optimizer,
    train_dataset: CacheDataset,
    args: DistillationTrainingArgs,
    training_callback: TrainingCallback = None,
):
    teacher_model.eval()
    teacher_model.freeze()
    model.train()

    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()

    if args.gradient_accumulation_steps != 1 and rank == 0:
        tqdm.write(
            "[distill] gradient_accumulation_steps > 1 detected; overriding to 1 for distillation phase."
        )
    args.gradient_accumulation_steps = 1

    iterator = iterate_online_dpo_batches(
        dataset=train_dataset,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        train=True,
    )

    def distill_loss(model, inputs, mask):
        student_logits = model(inputs).astype(mx.float32)
        teacher_logits = mx.stop_gradient(teacher_model(inputs).astype(mx.float32))

        student_log_probs = nn.log_softmax(student_logits[:, :-1, :], axis=-1)
        teacher_log_probs = nn.log_softmax(teacher_logits[:, :-1, :], axis=-1)
        teacher_probs = mx.exp(teacher_log_probs)

        targets = inputs[:, 1:]
        kl_per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(axis=-1)
        kl_sum = (kl_per_token * mask).sum()
        num_tokens = mask.sum()
        if num_tokens == 0:
            raise ValueError("No generated tokens available for distillation loss.")
        loss = kl_sum / num_tokens

        teacher_token_logps = mx.take_along_axis(
            teacher_log_probs, targets[..., None], axis=-1
        ).squeeze(-1)
        student_token_logps = mx.take_along_axis(
            student_log_probs, targets[..., None], axis=-1
        ).squeeze(-1)

        teacher_sum = (teacher_token_logps * mask).sum()
        student_sum = (student_token_logps * mask).sum()
        return loss, (num_tokens, teacher_sum, student_sum, kl_sum)

    loss_value_and_grad = nn.value_and_grad(model, distill_loss)

    loss_numerator = mx.array(0.0, dtype=mx.float32)
    total_tokens = mx.array(0.0, dtype=mx.float32)
    kl_total = mx.array(0.0, dtype=mx.float32)
    teacher_log_total = mx.array(0.0, dtype=mx.float32)
    student_log_total = mx.array(0.0, dtype=mx.float32)

    window_tokens = mx.array(0.0, dtype=mx.float32)
    window_time = 0.0

    pbar = tqdm(total=args.iters, desc="Distill", disable=rank != 0)
    it = 0
    tic = time.perf_counter()
    consecutive_empty = 0
    max_empty = 5
    while it < args.iters:
        batch = next(iterator)
        prepared = _prepare_distill_inputs(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_tokens=args.max_generation_tokens,
            temperature=args.distill_temperature,
        )
        if prepared is None:
            consecutive_empty += 1
            if consecutive_empty >= max_empty:
                raise RuntimeError(
                    "Failed to generate non-empty student completions after multiple attempts."
                )
            tic = time.perf_counter()
            continue
        consecutive_empty = 0

        inputs_np, prompt_lengths_np, lengths_np = prepared
        inputs = mx.array(inputs_np)
        prompt_lengths = mx.array(prompt_lengths_np)
        lengths = mx.array(lengths_np)
        mask = _compute_generation_mask(prompt_lengths, lengths, inputs.shape[1])

        (loss_value, metrics), grad = loss_value_and_grad(model, inputs, mask)
        grad = average_gradients(grad)
        optimizer.update(model, grad)

        num_tokens = metrics[0]
        teacher_sum = metrics[1]
        student_sum = metrics[2]
        kl_sum = metrics[3]

        loss_numerator += loss_value * num_tokens
        total_tokens += num_tokens
        kl_total += kl_sum
        teacher_log_total += teacher_sum
        student_log_total += student_sum
        window_tokens += num_tokens

        it += 1
        pbar.update(1)
        window_time += time.perf_counter() - tic

        if it % args.steps_per_report == 0 or it == args.iters:
            mx.eval(loss_numerator, total_tokens, kl_total, teacher_log_total, student_log_total, window_tokens)

            total_loss = mx.distributed.all_sum(loss_numerator, stream=mx.cpu).item()
            total_tok = mx.distributed.all_sum(total_tokens, stream=mx.cpu).item()
            total_kl = mx.distributed.all_sum(kl_total, stream=mx.cpu).item()
            total_teacher = mx.distributed.all_sum(teacher_log_total, stream=mx.cpu).item()
            total_student = mx.distributed.all_sum(student_log_total, stream=mx.cpu).item()
            window_tok = mx.distributed.all_sum(window_tokens, stream=mx.cpu).item()

            avg_loss = total_loss / max(total_tok, 1.0)
            avg_kl = total_kl / max(total_tok, 1.0)
            avg_teacher = total_teacher / max(total_tok, 1.0)
            avg_student = total_student / max(total_tok, 1.0)
            avg_tokens_per_sec = window_tok / max(window_time, 1e-6)
            lr_value = optimizer.learning_rate.item()

            if rank == 0:
                tqdm.write(
                    f"\nDistill iter {it}: "
                    f"loss {avg_loss:.4f}, "
                    f"kl/token {avg_kl:.4f}, "
                    f"teacher_logp {avg_teacher:.4f}, "
                    f"student_logp {avg_student:.4f}, "
                    f"lr {lr_value:.3e}, "
                    f"window_tok/s {avg_tokens_per_sec:.3f}"
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": avg_loss,
                    "learning_rate": lr_value,
                    "tokens_per_second": avg_tokens_per_sec,
                    "kl_per_token": avg_kl,
                    "teacher_logp": avg_teacher,
                    "student_logp": avg_student,
                }
                training_callback.on_train_loss_report(train_info)

            loss_numerator = mx.array(0.0, dtype=mx.float32)
            total_tokens = mx.array(0.0, dtype=mx.float32)
            kl_total = mx.array(0.0, dtype=mx.float32)
            teacher_log_total = mx.array(0.0, dtype=mx.float32)
            student_log_total = mx.array(0.0, dtype=mx.float32)
            window_tokens = mx.array(0.0, dtype=mx.float32)
            window_time = 0.0

        if it % args.steps_per_save == 0 and rank == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            tqdm.write(
                f"\n"
                f"Distill iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

        tic = time.perf_counter()

    if rank == 0:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(args.adapter_file), adapter_weights)
        tqdm.write(f"Saved final weights to {args.adapter_file}.")
