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
    max_tokens: int,
    temperature: float,
) -> List[List[int]]:
    sampler = make_sampler(
        temperature,
        top_p=1.0,
        min_p=0.0,
        min_tokens_to_keep=1,
        top_k=0,
        xtc_probability=0.0,
        xtc_threshold=0.0,
        xtc_special_tokens=tokenizer.encode("\n") + list(tokenizer.eos_token_ids),
    )

    completions: List[List[int]] = []
    for prompt_text in prompt_texts:
        completion = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        if isinstance(completion, str):
            completion_ids = tokenizer.encode(completion)
        else:
            completion_ids = completion

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
    completions = _sample_student_responses(
        model=model,
        tokenizer=tokenizer,
        prompt_texts=prompt_texts,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    sequences: List[List[int]] = []
    prompt_lengths: List[int] = []
    lengths: List[int] = []

    for prompt_ids, completion_ids in zip(prompts, completions):
        prompt_tokens = list(prompt_ids)
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
    if args.gradient_accumulation_steps != 1:
        raise ValueError("On-policy distillation currently requires gradient_accumulation_steps == 1.")

    teacher_model.eval()
    teacher_model.freeze()
    model.train()

    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()

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

    losses = 0.0
    n_tokens = 0.0
    kl_total = 0.0
    teacher_log_total = 0.0
    student_log_total = 0.0
    train_time = 0.0

    pbar = tqdm(total=args.iters, desc="Distill", disable=rank != 0)
    it = 0
    tic = time.perf_counter()
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
            continue

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

        losses += loss_value
        n_tokens += num_tokens
        kl_total += kl_sum
        teacher_log_total += teacher_sum
        student_log_total += student_sum
        mx.eval(losses, n_tokens, kl_total, teacher_log_total, student_log_total)

        it += 1
        pbar.update(1)
        train_time += time.perf_counter() - tic

        if it % args.steps_per_report == 0 or it == args.iters:
            total_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            total_tokens = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            total_kl = mx.distributed.all_sum(kl_total, stream=mx.cpu).item()
            total_teacher = mx.distributed.all_sum(teacher_log_total, stream=mx.cpu).item()
            total_student = mx.distributed.all_sum(student_log_total, stream=mx.cpu).item()

            avg_loss = total_loss / (it if it > 0 else 1)
            avg_tokens = total_tokens / max(train_time, 1e-6)
            avg_kl = total_kl / max(total_tokens, 1.0)
            avg_teacher = total_teacher / max(total_tokens, 1.0)
            avg_student = total_student / max(total_tokens, 1.0)
            lr_value = optimizer.learning_rate.item()

            if rank == 0:
                tqdm.write(
                    f"\nDistill iter {it}: "
                    f"loss {avg_loss:.4f}, "
                    f"kl/token {avg_kl:.4f}, "
                    f"teacher_logp {avg_teacher:.4f}, "
                    f"student_logp {avg_student:.4f}, "
                    f"lr {lr_value:.3e}, "
                    f"tok/s {avg_tokens:.3f}"
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": avg_loss,
                    "learning_rate": lr_value,
                    "tokens_per_second": avg_tokens,
                    "kl_per_token": avg_kl,
                    "teacher_logp": avg_teacher,
                    "student_logp": avg_student,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0.0
            n_tokens = 0.0
            kl_total = 0.0
            teacher_log_total = 0.0
            student_log_total = 0.0
            train_time = 0.0

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
