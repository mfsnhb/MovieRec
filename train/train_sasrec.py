from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.sasrec import SASRec, SASRecConfig
from utils.data_io import load_movie_feature_store, load_ratings
from utils.sasrec_data import (
    SASRecMappings,
    build_eval_examples,
    grouped_user_records,
    ndcg,
    sasrec_bce_loss,
    sasrec_collate,
    SASRecTrainDataset,
)
from utils.training_utils import Logger, ensure_dir, save_json, setup_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SASRec teacher on MovieLens chronology.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/funrec-movielens-1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/reranker/sasrec_prefix_teacher"))
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_mapping(path: Path, mappings: SASRecMappings, movie_titles: dict[str, str]) -> None:
    save_json(
        path,
        {
            "movie_id_to_index": mappings.movie_id_to_index,
            "index_to_movie_id": {str(index): movie_id for index, movie_id in mappings.index_to_movie_id.items()},
            "movie_titles": movie_titles,
            "user_ids": mappings.user_ids,
            "padding_index": 0,
        },
    )


def evaluate(model: SASRec, examples, batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"total": 0, "hr@1": 0, "hr@5": 0, "hr@10": 0, "ndcg@5": 0.0, "ndcg@10": 0.0}
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            input_ids = torch.tensor([example.input_ids for example in batch], dtype=torch.long, device=device)
            scores = model.score_all(input_ids)
            for row, example in enumerate(batch):
                row_scores = scores[row]
                for item_index in example.history_item_indices:
                    row_scores[item_index - 1] = -torch.inf
                target_col = example.target_item_index - 1
                target_score = row_scores[target_col]
                if not torch.isfinite(target_score):
                    rank = None
                else:
                    rank = int((row_scores > target_score).sum().item()) + 1
                totals["total"] += 1
                totals["hr@1"] += int(rank == 1)
                totals["hr@5"] += int(rank is not None and rank <= 5)
                totals["hr@10"] += int(rank is not None and rank <= 10)
                totals["ndcg@5"] += ndcg(rank) if rank is not None and rank <= 5 else 0.0
                totals["ndcg@10"] += ndcg(rank) if rank is not None and rank <= 10 else 0.0
    total = max(1, int(totals["total"]))
    return {
        "HR@1": totals["hr@1"] / total,
        "HR@5": totals["hr@5"] / total,
        "HR@10": totals["hr@10"] / total,
        "NDCG@5": totals["ndcg@5"] / total,
        "NDCG@10": totals["ndcg@10"] / total,
        "num_examples": total,
    }


def main() -> None:
    args = parse_args()
    logger = Logger()
    setup_seed(args.seed)
    ensure_dir(args.output_dir)
    save_json(args.output_dir / "training_args.json", vars(args))

    logger("1. Load MovieLens records")
    movie_features = load_movie_feature_store(args.raw_dir, {"movie_id", "title"})
    ratings_df = load_ratings(args.raw_dir, movie_features)
    records_by_user = grouped_user_records(ratings_df, args.max_users)
    mappings = SASRecMappings.from_movie_features(movie_features, records_by_user.keys())
    save_mapping(args.output_dir / "item_mapping.json", mappings, {movie_id: movie_features.title(movie_id) for movie_id in mappings.movie_ids})

    logger("2. Build SASRec train/eval data")
    train_dataset = SASRecTrainDataset(
        records_by_user,
        mappings,
        args.max_len,
        args.seed,
    )
    eval_examples = build_eval_examples(records_by_user, mappings, args.max_len, args.min_history)
    valid_examples = [example for example in eval_examples if example.split == "valid"]
    test_examples = [example for example in eval_examples if example.split == "test"]
    logger(f"Train examples: {len(train_dataset)}; valid: {len(valid_examples)}; test: {len(test_examples)}")

    logger("3. Train SASRec")
    device = torch.device(args.device)
    model = SASRec(
        SASRecConfig(
            num_items=mappings.num_items,
            max_len=args.max_len,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=sasrec_collate, num_workers=0)

    best_valid = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(loader, desc=f"sasrec epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            positive_ids = batch["positive_ids"].to(device)
            negative_ids = batch["negative_ids"].to(device)
            positive_scores = model.score_items(input_ids, positive_ids)
            negative_scores = model.score_items(input_ids, negative_ids)
            loss = sasrec_bce_loss(positive_scores, negative_scores, positive_ids)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite SASRec loss at epoch {epoch}: {loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
        valid_metrics = evaluate(model, valid_examples, args.eval_batch_size, device)
        epoch_record = {"epoch": epoch, "loss": float(np.mean(losses)), **{f"valid_{k}": v for k, v in valid_metrics.items()}}
        history.append(epoch_record)
        logger(json.dumps(epoch_record, ensure_ascii=False))
        valid_score = valid_metrics["NDCG@10"]
        if valid_score > best_valid:
            best_valid = valid_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "num_items": mappings.num_items}, args.output_dir / "model.pt")
        elif epoch - best_epoch >= args.patience:
            logger(f"Early stopping at epoch {epoch}; best epoch is {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("SASRec training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_examples, args.eval_batch_size, device)
    metrics = {"best_epoch": best_epoch, "best_valid_NDCG@10": best_valid, "test": test_metrics, "history": history}
    save_json(args.output_dir / "metrics.json", metrics)
    logger(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
