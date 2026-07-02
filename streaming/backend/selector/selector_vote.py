from __future__ import annotations

import random as _random
from typing import Any, Dict, List, Optional, Sequence

import torch
from streaming.backend.attention_metric import (
    ConditionMetricMode,
    stage2_score_by_view_from_attention_scores,
)
from streaming.backend.selector.selector import (
    SelectorVoteResult,
    Stage1SelectionResult,
    Stage2SelectionConfig,
    Stage2SelectionResult,
    TokenVoteMemory,
    ViewConditionSelector,
    ViewScoreBatch,
)


def view_score_batch_from_score_by_view(
    *,
    score_by_view: Dict[int, torch.Tensor],
    global_frame_indices: Sequence[int],
    frame_names: Sequence[str] | None = None,
    chunk_index: int | None = None,
    chunk_name: str | None = None,
    view_records: Sequence[Dict[str, Any]] | None = None,
) -> ViewScoreBatch:
    rows = []
    ordered_global_indices = []
    ordered_records = []
    for local_view_index, score in sorted(score_by_view.items()):
        local_view_index = int(local_view_index)
        score = score.detach().cpu().to(torch.float32).contiguous()
        row = score if score.dim() == 1 else score[local_view_index]
        rows.append(row.flatten().contiguous())
        global_frame_index = int(global_frame_indices[local_view_index])
        ordered_global_indices.append(global_frame_index)
        if view_records is not None:
            record = dict(view_records[local_view_index])
        else:
            record = {
                "global_frame_index": global_frame_index,
            }
            if frame_names is not None:
                record["frame_name"] = str(frame_names[local_view_index])
            if chunk_index is not None:
                record["chunk_index"] = int(chunk_index)
            if chunk_name is not None:
                record["chunk_name"] = str(chunk_name)
            record["local_view_index"] = local_view_index
        ordered_records.append(record)

    return ViewScoreBatch(
        scores=torch.stack(rows, dim=0).contiguous(),
        global_frame_indices=torch.tensor(ordered_global_indices, dtype=torch.long),
        view_records=ordered_records,
    )


def stage2_view_score_batch_from_attention_scores(
    *,
    attention_scores_by_view: Dict[int, torch.Tensor],
    candidates: Sequence[Dict[str, Any]],
    metric: ConditionMetricMode,
    patch_start: int,
    patch_end: int,
    kappa: float,
) -> ViewScoreBatch:
    score_by_view = stage2_score_by_view_from_attention_scores(
        attention_scores_by_view=attention_scores_by_view,
        metric=metric,
        patch_start=int(patch_start),
        patch_end=int(patch_end),
        kappa=float(kappa),
    )
    return view_score_batch_from_score_by_view(
        score_by_view=score_by_view,
        global_frame_indices=[
            int(candidates[int(view_idx)]["global_frame_index"])
            for view_idx in sorted(score_by_view)
        ],
        view_records=[dict(candidate) for candidate in candidates],
    )


def iter_stage2_candidate_batches(
    candidate_pool: Sequence[Dict[str, Any]],
    *,
    batch_size: int,
) -> Sequence[List[Dict[str, Any]]]:
    batches = []
    for start in range(0, len(candidate_pool), int(batch_size)):
        batches.append(
            [dict(view) for view in candidate_pool[start : start + int(batch_size)]]
        )
    return batches


def update_token_vote_memory(
    memory: TokenVoteMemory,
    batch: ViewScoreBatch,
    memory_depth: int,
    *,
    update_margin: float = 0.0,
) -> TokenVoteMemory:
    current_scores = (
        batch.scores.detach().cpu().to(torch.float32).transpose(0, 1).contiguous()
    )
    num_tokens = int(current_scores.shape[0])
    current_global_frame_indices = (
        batch.global_frame_indices.detach().cpu().to(torch.long).contiguous()
    )
    current_frame_table = current_global_frame_indices.unsqueeze(0).expand(
        num_tokens, -1
    )

    if memory.score is None:
        old_scores = torch.full(
            (num_tokens, int(memory_depth)), float("-inf"), dtype=torch.float32
        )
        old_global_frame_index = torch.full(
            (num_tokens, int(memory_depth)), -1, dtype=torch.long
        )
    else:
        old_scores = memory.score.detach().cpu().to(torch.float32).contiguous()
        old_global_frame_index = (
            memory.global_frame_index.detach().cpu().to(torch.long).contiguous()
        )

    view_records = {
        int(global_frame_index): dict(record)
        for global_frame_index, record in memory.view_records.items()
    }
    for position, global_frame_index in enumerate(current_global_frame_indices.tolist()):
        if batch.view_records:
            record = dict(batch.view_records[position])
        else:
            record = {"global_frame_index": int(global_frame_index)}
        view_records[int(global_frame_index)] = record

    merged_scores = torch.cat([old_scores, current_scores], dim=1)
    merged_global_frame_index = torch.cat(
        [old_global_frame_index, current_frame_table], dim=1
    )
    top_scores = torch.full(
        (num_tokens, int(memory_depth)), float("-inf"), dtype=torch.float32
    )
    top_global_frame_indices = torch.full(
        (num_tokens, int(memory_depth)), -1, dtype=torch.long
    )
    for token_idx in range(num_tokens):
        best_by_frame: dict[int, float] = {}
        for slot_idx in range(int(merged_global_frame_index.shape[1])):
            frame_index = int(merged_global_frame_index[token_idx, slot_idx].item())
            if frame_index < 0:
                continue
            value = float(merged_scores[token_idx, slot_idx].item())
            previous = best_by_frame.get(frame_index)
            if previous is None or value > previous + float(update_margin):
                best_by_frame[frame_index] = value

        ranked = sorted(
            best_by_frame.items(),
            key=lambda item: (-float(item[1]), int(item[0])),
        )[: int(memory_depth)]
        for rank, (frame_index, value) in enumerate(ranked):
            top_scores[token_idx, rank] = float(value)
            top_global_frame_indices[token_idx, rank] = int(frame_index)

    return TokenVoteMemory(
        score=top_scores.contiguous(),
        global_frame_index=top_global_frame_indices.contiguous(),
        view_records=view_records,
    )


def token_vote_report(
    memory: TokenVoteMemory,
    *,
    metric: ConditionMetricMode,
) -> List[Dict[str, Any]]:
    metric = ConditionMetricMode(metric)
    if memory.score is None or memory.global_frame_index is None:
        return []

    valid_mask = memory.global_frame_index >= 0
    if not bool(valid_mask.any()):
        return []

    flat_global_frame_indices = memory.global_frame_index[valid_mask].reshape(-1)
    flat_scores = memory.score[valid_mask].reshape(-1)
    total_slots = int(flat_scores.numel())
    mean_key, min_key, max_key = metric.best_summary_keys

    vote_rows = []
    for global_frame_index in sorted(
        int(value) for value in flat_global_frame_indices.unique().tolist()
    ):
        vote_mask = flat_global_frame_indices == int(global_frame_index)
        record = dict(memory.view_records.get(global_frame_index, {}))
        record.setdefault("global_frame_index", int(global_frame_index))
        slot_count = int(vote_mask.sum().item())
        vote_rows.append(
            {
                **record,
                "global_frame_index": int(global_frame_index),
                "token_count": slot_count,
                "token_fraction": float(slot_count / max(1, total_slots)),
                "slot_count": slot_count,
                "slot_fraction": float(slot_count / max(1, total_slots)),
                mean_key: float(flat_scores[vote_mask].mean().item()),
                min_key: float(flat_scores[vote_mask].min().item()),
                max_key: float(flat_scores[vote_mask].max().item()),
            }
        )

    vote_rows = sorted(
        vote_rows,
        key=lambda row: (
            -int(row["token_count"]),
            -float(row[mean_key]),
            int(row["global_frame_index"]),
        ),
    )
    return [
        {**row, "rank": int(rank)}
        for rank, row in enumerate(vote_rows, start=1)
    ]


def select_token_vote_views(
    memory: TokenVoteMemory,
    topk: int,
    *,
    metric: ConditionMetricMode,
) -> List[Dict[str, Any]]:
    return token_vote_report(memory, metric=metric)[: int(topk)]


def _seen_frame_indices(memory: TokenVoteMemory) -> List[int]:
    return sorted(int(k) for k in memory.view_records.keys())


# ---------------------------------------------------------------------------
# Task-32: divergence-regularized selection (Task-31 proposals).
# These operate on a per-view × per-element evidence matrix M [V, E] (E = tokens
# at stage-1, latents at stage-2). rel_v = row-sum; redundancy = cosine(M_u, M_v).
#   facility : submodular token/latent coverage (residual greedy, 1-1/e).
#   mmr      : argmax_v [ rel_v - lambda * max_{u in S} cos(M_v, M_u) ].
#   dpp      : greedy log-det MAP of L = diag(q) S diag(q), q=sqrt(rel_norm), S=cosine.
# ---------------------------------------------------------------------------
MMR_LAMBDA = 1.0
DPP_EPS = 1e-9


def _view_token_matrix(memory: TokenVoteMemory):
    """Reconstruct a sparse per-view × per-token evidence matrix from the token
    memory (each token keeps its top-`memory_depth` views). Returns (M, gfis)."""
    gf = memory.global_frame_index
    sc = memory.score
    if gf is None or sc is None:
        return None, []
    num_tokens, depth = int(gf.shape[0]), int(gf.shape[1])
    gfis = sorted(int(v) for v in gf[gf >= 0].unique().tolist())
    pos = {g: i for i, g in enumerate(gfis)}
    M = torch.zeros((len(gfis), num_tokens), dtype=torch.float32)
    for t in range(num_tokens):
        for s in range(depth):
            v = int(gf[t, s].item())
            if v < 0:
                continue
            val = float(sc[t, s].item())
            i = pos[v]
            if val > M[i, t]:
                M[i, t] = val
    return M, gfis


def select_views_by_matrix(M, gfis: Sequence[int], topk: int, strategy: str,
                           mmr_lambda: float = MMR_LAMBDA) -> List[int]:
    """Return selected global_frame_indices using a divergence-regularized objective."""
    V = int(M.shape[0])
    k = min(int(topk), V)
    if k <= 0:
        return []
    rel = M.sum(dim=1)  # [V]
    if strategy == "facility":
        covered = torch.zeros(M.shape[1], dtype=torch.float32)
        chosen: List[int] = []
        avail = set(range(V))
        for _ in range(k):
            best, best_gain = -1, float("-inf")
            for v in avail:
                gain = float(torch.clamp(M[v] - covered, min=0.0).sum().item())
                if gain > best_gain:
                    best_gain, best = gain, v
            if best < 0:
                break
            chosen.append(best); avail.discard(best)
            covered = torch.maximum(covered, M[best])
        return [int(gfis[i]) for i in chosen]
    # cosine similarity matrix (shared by mmr + dpp)
    norm = M.norm(dim=1).clamp(min=DPP_EPS)
    F = M / norm.unsqueeze(1)
    S = (F @ F.t()).clamp(-1.0, 1.0)  # [V,V], diag≈1
    rel_n = rel / rel.max().clamp(min=DPP_EPS)
    if strategy == "mmr":
        chosen = [int(torch.argmax(rel_n).item())]
        while len(chosen) < k:
            best, best_score = -1, float("-inf")
            for v in range(V):
                if v in chosen:
                    continue
                sim_max = float(S[v, chosen].max().item())
                score = float(rel_n[v].item()) - float(mmr_lambda) * sim_max
                if score > best_score:
                    best_score, best = score, v
            if best < 0:
                break
            chosen.append(best)
        return [int(gfis[i]) for i in chosen]
    if strategy == "dpp":
        q = torch.sqrt(rel_n.clamp(min=DPP_EPS))
        L = (q.unsqueeze(1) * q.unsqueeze(0)) * S + DPP_EPS * torch.eye(V)
        chosen: List[int] = []
        for _ in range(k):
            best, best_gain = -1, float("-inf")
            for v in range(V):
                if v in chosen:
                    continue
                idx = chosen + [v]
                sub = L[idx][:, idx]
                sign, logdet = torch.slogdet(sub)
                gain = float(logdet.item()) if sign.item() > 0 else float("-inf")
                if gain > best_gain:
                    best_gain, best = gain, v
            if best < 0:
                break
            chosen.append(best)
        return [int(gfis[i]) for i in chosen]
    raise ValueError(f"unknown matrix strategy: {strategy}")


ADVANCED_STRATEGIES = ("facility", "mmr", "dpp")


def _va_div_select(report: Sequence[Dict[str, Any]], k: int, count_key: str,
                   pool: Sequence[int], div_lambda: float) -> List[Dict[str, Any]]:
    """Task-42 distance-aware VA selection: greedy frame-distance MMR over the VA vote mass.
    rel_v = count_key (vote/latent mass); overlap(v,u) = max(0, 1 - |gfi_v-gfi_u|/span) (frame-distance
    on the spiral trajectory ~ camera-angle distance). score_v = rel_norm_v - lambda*max_{u in S} overlap.
    Greedy: seed with the top-mass view (== vote's first pick), then add argmax score. lambda=0 -> vote."""
    rows = list(report)
    if len(rows) <= k:
        return [{**r} for r in rows[:k]]
    gfi = {id(r): int(r["global_frame_index"]) for r in rows}
    rel = {id(r): float(r.get(count_key, 0) or 0.0) for r in rows}
    rmax = max(rel.values()) or 1.0
    lo, hi = min(pool), max(pool)
    span = max(int(hi) - int(lo), 1)
    # Redundancy kernel is LOCAL, scaled to the ideal uniform spacing span/k: two views are "redundant"
    # only if closer than uniform coverage would place them. With a global 1-d/span kernel and a wide pool
    # every pair looks ~equally overlapping (~0.9), so lambda<=0.2 cannot punish adjacency specifically;
    # span/k makes overlap=1 for coincident, 0 at/ beyond ideal spacing -> lambda in [0,0.2] de-clusters.
    win = max(span / float(max(k, 1)), 1.0)
    def overlap(a, b) -> float:
        return max(0.0, 1.0 - abs(gfi[id(a)] - gfi[id(b)]) / win)
    chosen = [sorted(rows, key=lambda r: (-rel[id(r)], gfi[id(r)]))[0]]
    chosen_ids = {id(chosen[0])}
    while len(chosen) < k:
        best, best_s = None, float("-inf")
        for r in rows:
            if id(r) in chosen_ids:
                continue
            omax = max(overlap(r, c) for c in chosen)
            s = rel[id(r)] / rmax - float(div_lambda) * omax
            if s > best_s or (s == best_s and (best is None or gfi[id(r)] < gfi[id(best)])):
                best_s, best = s, r
        if best is None:
            break
        chosen.append(best); chosen_ids.add(id(best))
    return [{**r} for r in sorted(chosen, key=lambda r: gfi[id(r)])[:k]]


def apply_selection_strategy(
    report: Sequence[Dict[str, Any]],
    *,
    topk: int,
    strategy: str,
    seed: int,
    count_key: str,
    pool_indices: Sequence[int],
    div_lambda: float = 0.0,
) -> List[Dict[str, Any]]:
    """Shared Task-30 re-selection over a vote/latent report (used by stage-1 AND stage-2).
    `count_key` is the per-view evidence count ("token_count" / "latent_count");
    `pool_indices` is the full seen/candidate frame-index pool (for random + bin spans).
      - "vote"    -> top-k by count (default).
      - "random"  -> k random from the pool.
      - "diverse" -> split the index span (≈azimuth on the spiral) into k bins, take the
                     highest-count view per bin; back-fill empties by remaining top-count."""
    k = int(topk)
    rows_by_gfi = {int(r["global_frame_index"]): r for r in report}
    pool = sorted({int(i) for i in pool_indices}) or sorted(rows_by_gfi)
    if strategy == "random":
        kk = min(k, len(pool))
        chosen = sorted(_random.Random(int(seed)).sample(pool, kk)) if pool else []
        out: List[Dict[str, Any]] = []
        for gfi in chosen:
            out.append({**rows_by_gfi[gfi]} if gfi in rows_by_gfi
                       else {"global_frame_index": int(gfi), count_key: 0})
        return out
    if strategy == "uniform":
        # vote-agnostic pure coverage: k evenly-spaced views across the seen pool
        # (position-spaced ⇒ ≈uniform azimuth on the spiral). Ignores evidence entirely.
        n = len(pool)
        kk = min(k, n)
        if kk <= 0:
            return []
        if kk == 1:
            picks = [pool[n // 2]]
        else:
            picks = sorted({pool[round(i * (n - 1) / (kk - 1))] for i in range(kk)})
        gi = 0
        while len(picks) < kk:  # fill if rounding collided
            if pool[gi] not in picks:
                picks.append(pool[gi])
            gi += 1
        picks = sorted(picks)[:k]
        return [({**rows_by_gfi[g]} if g in rows_by_gfi else {"global_frame_index": int(g), count_key: 0})
                for g in picks]
    if strategy == "diverse":
        if len(report) <= k:
            return [{**r} for r in report[:k]]
        lo, hi = min(pool), max(pool)
        span = (hi - lo + 1)
        used: set[int] = set()
        picked: List[Dict[str, Any]] = []
        for b in range(k):
            b_lo = lo + (span * b) // k
            b_hi = lo + (span * (b + 1)) // k
            cand = [r for r in report
                    if b_lo <= int(r["global_frame_index"]) < b_hi
                    and int(r["global_frame_index"]) not in used]
            if cand:
                best = max(cand, key=lambda r: (int(r.get(count_key, 0)), -int(r["global_frame_index"])))
                picked.append(best)
                used.add(int(best["global_frame_index"]))
        for r in report:  # back-fill empty bins by remaining highest-count
            if len(picked) >= k:
                break
            if int(r["global_frame_index"]) not in used:
                picked.append(r)
                used.add(int(r["global_frame_index"]))
        return [{**r} for r in sorted(picked, key=lambda r: int(r["global_frame_index"]))[:k]]
    if strategy == "va_div":
        return _va_div_select(report, k, count_key, pool, div_lambda)
    return [{**r} for r in report[:k]]


def selector_vote(
    memory: TokenVoteMemory,
    batch: ViewScoreBatch,
    topk: int,
    memory_depth: int,
    metric: ConditionMetricMode,
    update_margin: float = 0.0,
    strategy: str = "vote",
    random_seed: int = 0,
    div_lambda: float = 0.0,
) -> SelectorVoteResult:
    updated_memory = update_token_vote_memory(
        memory,
        batch,
        memory_depth,
        update_margin=float(update_margin),
    )
    report = token_vote_report(updated_memory, metric=metric)
    if strategy in ("random", "diverse", "uniform", "va_div"):
        selected_views = apply_selection_strategy(
            report, topk=topk, strategy=strategy, seed=random_seed,
            count_key="token_count", pool_indices=_seen_frame_indices(updated_memory),
            div_lambda=div_lambda,
        )
    elif strategy in ADVANCED_STRATEGIES:
        M, gfis = _view_token_matrix(updated_memory)
        if M is None or len(gfis) <= int(topk):
            selected_views = report[: int(topk)]
        else:
            chosen = set(select_views_by_matrix(M, gfis, topk, strategy))
            rows = {int(r["global_frame_index"]): r for r in report}
            selected_views = [({**rows[g]} if g in rows else {"global_frame_index": int(g), "token_count": 0})
                              for g in sorted(chosen)]
    else:
        selected_views = report[: int(topk)]
    return SelectorVoteResult(memory=updated_memory, selected_views=selected_views)


def select_stage2_views_from_score_batches(
    *,
    score_batches: Sequence[ViewScoreBatch],
    topk: int,
    metric: ConditionMetricMode,
    strategy: str = "vote",
    random_seed: int = 0,
    div_lambda: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    rows = []
    records = []
    for batch in score_batches:
        for position, global_frame_index in enumerate(batch.global_frame_indices.tolist()):
            rows.append(batch.scores[position].detach().cpu().to(torch.float32).flatten())
            if batch.view_records:
                record = dict(batch.view_records[position])
            else:
                record = {"global_frame_index": int(global_frame_index)}
            record["global_frame_index"] = int(global_frame_index)
            records.append(record)

    metric_matrix = torch.stack(rows, dim=0)
    best_positions = torch.argmax(metric_matrix, dim=0)
    total_latents = int(best_positions.numel())
    mean_key = f"mean_{ConditionMetricMode(metric).value}"
    min_key = f"min_{ConditionMetricMode(metric).value}"
    max_key = f"max_{ConditionMetricMode(metric).value}"
    report = []
    for position, record in enumerate(records):
        latent_mask = best_positions == int(position)
        latent_count = int(latent_mask.sum().item())
        if latent_count == 0:
            continue
        winning_scores = metric_matrix[position, latent_mask]
        report.append(
            {
                **record,
                "latent_count": latent_count,
                "latent_fraction": float(latent_count / total_latents),
                mean_key: float(winning_scores.mean().item()),
                min_key: float(winning_scores.min().item()),
                max_key: float(winning_scores.max().item()),
            }
        )

    report = sorted(
        report,
        key=lambda row: (
            -int(row["latent_count"]),
            -float(row[mean_key]),
            int(row["global_frame_index"]),
        ),
    )
    report = [
        {**row, "rank": int(rank)}
        for rank, row in enumerate(report, start=1)
    ]
    if strategy in ("random", "diverse", "uniform", "va_div"):
        pool_indices = [int(r["global_frame_index"]) for r in records]
        selected = apply_selection_strategy(
            report, topk=topk, strategy=strategy, seed=random_seed,
            count_key="latent_count", pool_indices=pool_indices,
            div_lambda=div_lambda,
        )
    elif strategy in ADVANCED_STRATEGIES:
        gfis = [int(r["global_frame_index"]) for r in records]
        rep_by_gfi = {int(r["global_frame_index"]): r for r in report}
        if len(gfis) <= int(topk):
            selected = [dict(row) for row in report[: int(topk)]]
        else:
            chosen = set(select_views_by_matrix(metric_matrix, gfis, topk, strategy))
            selected = [({**rep_by_gfi[g]} if g in rep_by_gfi else {"global_frame_index": int(g), "latent_count": 0})
                        for g in sorted(chosen)]
    else:
        selected = [dict(row) for row in report[: int(topk)]]
    return selected, report, len(records)


def run_stage2_vote_selection(
    *,
    score_batches: Sequence[ViewScoreBatch],
    topk: int,
    memory_depth: int,
    metric: ConditionMetricMode,
    attention_view_count: int,
    collector_steps: Sequence[int],
    warnings: Sequence[str],
    strategy: str = "vote",
    random_seed: int = 0,
    div_lambda: float = 0.0,
) -> Stage2SelectionResult:
    _ = memory_depth
    selected_views, vote_report, candidate_pool_size = select_stage2_views_from_score_batches(
        score_batches=score_batches,
        topk=topk,
        metric=metric,
        strategy=strategy,
        random_seed=random_seed,
        div_lambda=div_lambda,
    )

    return Stage2SelectionResult(
        selected_views=selected_views,
        warnings=list(warnings),
        candidate_pool_size=int(candidate_pool_size),
        attention_view_count=int(attention_view_count),
        collector_steps=[int(step) for step in collector_steps],
        view_vote_report=vote_report,
    )


class TokenVoteSelector(ViewConditionSelector):
    @property
    def needs_ss_warmup(self) -> bool:
        return True

    def stage1_select(
        self,
        *,
        chunk_spec: Dict[str, Any],
        warmup_chunk_spec: Dict[str, Any],
        warmup: Optional[Dict[str, Any]],
        score_by_view: Optional[Dict[int, torch.Tensor]],
        frame_names: Sequence[str],
        chunk_index: int,
        chunk_name: str,
    ) -> Stage1SelectionResult:
        if score_by_view is None:
            raise ValueError("TOKEN_VOTE selection requires per-view token scores.")

        batch = view_score_batch_from_score_by_view(
            score_by_view=score_by_view,
            global_frame_indices=warmup_chunk_spec["global_frame_indices"],
            frame_names=frame_names,
            chunk_index=chunk_index,
            chunk_name=chunk_name,
        )
        vote_result = selector_vote(
            self.state.token_memory,
            batch,
            topk=self.config.topk,
            memory_depth=self.config.memory_depth,
            metric=self.config.metric,
            update_margin=self.config.token_vote_update_margin,
            strategy=getattr(self.config, "selection_strategy", "vote"),
            random_seed=getattr(self.config, "selection_random_seed", 0),
            div_lambda=getattr(self.config, "selection_div_lambda", 0.0),
        )
        self.state.token_memory = vote_result.memory

        return Stage1SelectionResult(
            selected_views=vote_result.selected_views,
            warnings=[],
            metadata={"selection_strategy": getattr(self.config, "selection_strategy", "vote"),
                      "selection_div_lambda": float(getattr(self.config, "selection_div_lambda", 0.0))},
        )

    def stage2_select(
        self,
        *,
        seen_chunk_specs: Sequence[Dict[str, Any]],
        chunk_index: int,
        chunk_name: str,
        score_batches: Sequence[ViewScoreBatch],
        attention_view_count: int,
        collector_steps: Sequence[int],
        warnings: Sequence[str],
        config: Stage2SelectionConfig,
    ) -> Stage2SelectionResult:
        return run_stage2_vote_selection(
            score_batches=score_batches,
            topk=config.topk,
            memory_depth=self.config.memory_depth,
            metric=config.metric,
            attention_view_count=attention_view_count,
            collector_steps=collector_steps,
            warnings=warnings,
            strategy=getattr(self.config, "selection_strategy", "vote"),
            random_seed=getattr(self.config, "selection_random_seed", 0),
            div_lambda=getattr(self.config, "selection_div_lambda", 0.0),
        )
