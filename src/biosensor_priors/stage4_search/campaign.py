"""Multi-round retrospective evolutionary campaigns (BO-EVO evaluation protocol)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biosensor_priors.common.config import REPO_ROOT, load_yaml, resolve_path
from biosensor_priors.common.provenance import write_manifest
from biosensor_priors.stage3_surrogate.features import FeatureBuilder
from biosensor_priors.stage3_surrogate.surrogate import FusedSurrogate
from biosensor_priors.stage4_search.landscape import build_landscape_view, hamming
from biosensor_priors.stage4_search.policy import build_search_policies


def _load_master(root: Path) -> pd.DataFrame:
    """Load the processed experiment master table from disk.

    Parameters
    ----------
    root : Path
        Repository root containing ``data/processed/``.

    Returns
    -------
    pd.DataFrame
        Experiment master table preferring pickle over parquet when both exist.
    """
    pkl = root / "data" / "processed" / "experiment_master.pkl"
    if pkl.exists():
        return pd.read_pickle(pkl)
    return pd.read_parquet(root / "data" / "processed" / "experiment_master.parquet")


def _policies(search_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Instantiate search policies for retrospective campaign benchmarks."""
    return build_search_policies(search_cfg, seed)


def choose_starting_indices(
    y: np.ndarray,
    sequences: list[str],
    *,
    initial_n: int,
    wt_fitness: float = 0.1,
    window: float = 0.2,
    random_seed: int = 0,
) -> list[int]:
    """Select diverse starting indices near wild-type fitness for campaigns.

    Approximates the paper starting-sequence protocol: filter to a fitness band
    above zero and below ``window``, bin by fitness, diversify by Hamming
    distance to the global optimum, then sample.

    Parameters
    ----------
    y : np.ndarray
        Measured fitness values for all landscape constructs.
    sequences : list of str
        Variable-site sequence strings aligned with ``y``.
    initial_n : int
        Number of starting constructs to select.
    wt_fitness : float, optional
        Reference wild-type fitness (default 0.1); lower band bound is 0.
    window : float, optional
        Upper fitness bound for the starting band (default 0.2).
    random_seed : int, optional
        RNG seed for tie-breaking and fill sampling (default 0).

    Returns
    -------
    list of int
        Landscape row indices chosen as round-0 observations.
    """
    rng = np.random.default_rng(random_seed)
    global_opt = int(np.argmax(y))
    lo = 0.0
    hi = float(window)
    # Prefer variants near the configured WT fitness band.
    band = np.where((y > lo) & (y <= hi))[0]
    if len(band) < initial_n:
        # Fallback: lowest-fitness initial_n (cool start alternative)
        return list(np.argsort(y)[:initial_n])

    # 5 fitness bins within (0, window)
    edges = np.linspace(lo, hi, 6)
    chosen: list[int] = []
    for b in range(5):
        in_bin = [i for i in band if edges[b] < y[i] <= edges[b + 1]]
        if not in_bin:
            continue
        # Group by Hamming distance to global optima
        by_dist: dict[int, list[int]] = {}
        for i in in_bin:
            d = hamming(sequences[i], sequences[global_opt])
            by_dist.setdefault(d, []).append(i)
        for dist in sorted(by_dist):
            pick = int(rng.choice(by_dist[dist]))
            if pick not in chosen:
                chosen.append(pick)
            if len(chosen) >= initial_n:
                return chosen[:initial_n]
    # Fill remaining randomly from band
    remain = [i for i in band if i not in chosen]
    rng.shuffle(remain)
    chosen.extend(remain)
    return chosen[:initial_n]


@dataclass
class CampaignResult:
    round_table: pd.DataFrame
    selected_table: pd.DataFrame


def run_single_campaign(
    *,
    algorithm: str,
    landscape: pd.DataFrame,
    policy,
    encoding: str,
    initial: list[int],
    n_rounds: int,
    batch_size: int,
    random_seed: int,
    top_quantile: float = 0.05,
) -> CampaignResult:
    """Simulate one multi-round active-learning campaign on a fixed landscape.

    Parameters
    ----------
    algorithm : str
        Strategy name recorded in output tables.
    landscape : pd.DataFrame
        Full measured fitness landscape with ``fitness`` and ``construct_id``.
    policy
        Search policy implementing ``propose``.
    encoding : str
        Feature encoding passed to the per-round surrogate refit.
    initial : list of int
        Landscape indices revealed before round 1.
    n_rounds : int
        Number of acquisition rounds after the initial reveal.
    batch_size : int
        Constructs proposed and revealed per round.
    random_seed : int
        Base seed for surrogate refits across rounds.
    top_quantile : float, optional
        Quantile defining campaign success (default 0.05).

    Returns
    -------
    CampaignResult
        Per-round metrics and selected-construct tables for the simulation.
    """
    work = landscape.reset_index(drop=True).copy()
    y = work["fitness"].to_numpy(dtype=float)
    view = build_landscape_view(work)
    sequences = view.sequences
    n = len(work)
    observed = set(initial)
    available = set(range(n)) - observed
    global_best = float(np.max(y))
    top_cutoff = float(np.quantile(y, 1.0 - top_quantile))

    round_rows = []
    selected_rows = []

    def _summary(round_no: int, batch_idx: list[int] | None = None) -> dict[str, Any]:
        """Compute cumulative campaign metrics for one round.

        Parameters
        ----------
        round_no : int
            Round index (0 for initial reveal).
        batch_idx : list of int or None, optional
            Landscape indices revealed this round; defaults to all observed.

        Returns
        -------
        dict
            Metric row with batch and cumulative fitness statistics.
        """
        obs_list = sorted(observed)
        cum = float(np.max(y[obs_list]))
        batch_fit = y[batch_idx] if batch_idx else y[obs_list]
        return {
            "Algorithm": algorithm,
            "Round": round_no,
            "N_measured": len(observed),
            "Batch_mean_fitness": float(np.mean(batch_fit)),
            "Batch_max_fitness": float(np.max(batch_fit)),
            "Cumulative_best_fitness": cum,
            "Regret": global_best - cum,
            "Reached_global_best": int(np.isclose(cum, global_best)),
            "Reached_top_quantile": int(cum >= top_cutoff),
            "Success": int(cum >= top_cutoff or np.isclose(cum, global_best)),
        }

    round_rows.append(_summary(0, list(initial)))
    for i in initial:
        selected_rows.append(
            {
                "Algorithm": algorithm,
                "Round": 0,
                "construct_id": str(work.iloc[i].get("construct_id", i)),
                "fitness": float(y[i]),
            }
        )

    for round_no in range(1, n_rounds + 1):
        if not available:
            break
        train = work.iloc[sorted(observed)].copy()
        pool = work.iloc[sorted(available)].copy()
        surrogate = FusedSurrogate(
            kind="gp_zero_mean",  # paper baseline uses plain GPR on fitness
            use_confidence_weighting=False,
            random_state=random_seed + round_no,
            feature_builder=FeatureBuilder(encoding=encoding, include_physics=False),  # type: ignore[arg-type]
            multi_output=False,
            kernel="matern52",
            version_intercept=False,
            fit_physics_alpha=False,
        )
        surrogate.fit(train, train["fitness"].to_numpy(dtype=float))
        batch_df = policy.propose(train, pool, surrogate, batch_size)
        # Map proposed rows back to landscape indices via construct_id
        if "construct_id" in batch_df.columns and "construct_id" in work.columns:
            id_to_idx = {str(cid): i for i, cid in enumerate(work["construct_id"].astype(str))}
            batch_idx = [
                id_to_idx[str(cid)]
                for cid in batch_df["construct_id"]
                if str(cid) in id_to_idx and id_to_idx[str(cid)] in available
            ]
        else:
            # fallback: match on sequence
            seq_to_idx = {seq: i for i, seq in enumerate(sequences)}
            batch_idx = []
            for seq in batch_df.get("sequence", []):
                i = seq_to_idx.get(str(seq))
                if i is not None and i in available:
                    batch_idx.append(i)

        # Ensure we always reveal up to batch_size new points
        if len(batch_idx) < batch_size:
            remain = [i for i in sorted(available) if i not in batch_idx]
            batch_idx.extend(remain[: batch_size - len(batch_idx)])

        for i in batch_idx:
            observed.add(i)
            available.discard(i)
            selected_rows.append(
                {
                    "Algorithm": algorithm,
                    "Round": round_no,
                    "construct_id": str(work.iloc[i].get("construct_id", i)),
                    "fitness": float(y[i]),
                }
            )
        round_rows.append(_summary(round_no, batch_idx))

    return CampaignResult(pd.DataFrame(round_rows), pd.DataFrame(selected_rows))


def summarize_campaigns(round_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize multi-repeat campaign results in paper-style aggregate metrics.

    Parameters
    ----------
    round_table : pd.DataFrame
        Per-round metrics from :func:`run_single_campaign`, optionally with a
        ``Repeat`` column.

    Returns
    -------
    pd.DataFrame
        One row per algorithm with success ratio and final cumulative statistics.
    """
    rows = []
    for algo, group in round_table.groupby("Algorithm"):
        finals = group[group["Round"] == group["Round"].max()]
        rows.append(
            {
                "Algorithm": algo,
                "Success_ratio": float(finals["Success"].mean()) if len(finals) else np.nan,
                "Final_cumulative_best_mean": float(finals["Cumulative_best_fitness"].mean()),
                "Final_cumulative_best_std": float(finals["Cumulative_best_fitness"].std(ddof=0)),
                "Final_batch_max_mean": float(finals["Batch_max_fitness"].mean()),
                "Final_batch_mean_mean": float(finals["Batch_mean_fitness"].mean()),
                "N_simulations": int(finals["Repeat"].nunique()) if "Repeat" in finals.columns else int(len(finals)),
            }
        )
    return pd.DataFrame(rows)


def run_campaign_benchmark(
    *,
    repo_root: Path | None = None,
    n_repeats: int | None = None,
    n_rounds: int | None = None,
    batch_size: int | None = None,
    initial_n: int | None = None,
    encoding: str | None = None,
) -> dict[str, Any]:
    """Run paired multi-round retrospective campaigns across search strategies.

    Parameters
    ----------
    repo_root : Path or None, optional
        Repository root; defaults to configured ``REPO_ROOT``.
    n_repeats : int or None, optional
        Number of independent campaign repeats; defaults to ``search.yaml``.
    n_rounds : int or None, optional
        Acquisition rounds per campaign; defaults to config.
    batch_size : int or None, optional
        Batch size per round; defaults to config.
    initial_n : int or None, optional
        Number of starting constructs; defaults to config.
    encoding : str or None, optional
        Surrogate feature encoding; defaults to config.

    Returns
    -------
    dict
        Keys include ``round_table``, ``selected_table``, ``summary``,
        ``manifest_path``, and ``output_dir``.

    Raises
    ------
    RuntimeError
        If the labeled landscape is too small for the requested campaign size.
    """
    root = repo_root or REPO_ROOT
    pipeline = load_yaml(root / "configs" / "pipeline.yaml")
    search_cfg = load_yaml(root / "configs" / "search.yaml")
    camp = search_cfg.get("campaign", {})
    seed = int(pipeline.get("random_seed", 42))

    n_repeats = int(n_repeats if n_repeats is not None else camp.get("n_repeats", 5))
    n_rounds = int(n_rounds if n_rounds is not None else camp.get("n_rounds", 5))
    batch_size = int(batch_size if batch_size is not None else search_cfg.get("batch_size", 8))
    initial_n = int(initial_n if initial_n is not None else camp.get("initial_n", 8))
    encoding = str(encoding if encoding is not None else camp.get("encoding", "hybrid"))
    strategies = list(search_cfg.get("strategies", ["random", "adalead", "mcmc", "bo"]))

    master = _load_master(root)
    landscape = master[master["fitness"].notna()].reset_index(drop=True).copy()
    if len(landscape) < initial_n + batch_size:
        raise RuntimeError("Not enough fitness-labeled constructs for a campaign.")

    view = build_landscape_view(landscape)
    y = landscape["fitness"].to_numpy(dtype=float)

    all_rounds = []
    all_selected = []
    for repeat in range(n_repeats):
        initial = choose_starting_indices(
            y,
            view.sequences,
            initial_n=initial_n,
            random_seed=seed + 17 * repeat,
        )
        policies = _policies(search_cfg, seed + repeat)
        for algo in strategies:
            result = run_single_campaign(
                algorithm=algo,
                landscape=landscape,
                policy=policies[algo],
                encoding=encoding,
                initial=initial,
                n_rounds=n_rounds,
                batch_size=batch_size,
                random_seed=seed + 100 * repeat,
            )
            rt = result.round_table.copy()
            rt["Repeat"] = repeat
            st = result.selected_table.copy()
            st["Repeat"] = repeat
            all_rounds.append(rt)
            all_selected.append(st)

    round_table = pd.concat(all_rounds, ignore_index=True)
    selected_table = pd.concat(all_selected, ignore_index=True)
    summary = summarize_campaigns(round_table)

    out_dir = resolve_path(pipeline["paths"]["outputs"], root) / "stage4_campaigns"
    out_dir.mkdir(parents=True, exist_ok=True)
    round_table.to_csv(out_dir / "round_metrics.csv", index=False)
    selected_table.to_csv(out_dir / "selected_constructs.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    manifest = write_manifest(
        resolve_path(pipeline["paths"]["manifests"], root) / "stage4_campaign_manifest.json",
        stage="stage4_campaign",
        inputs={"n_landscape": int(len(landscape)), "encoding": encoding},
        parameters={
            "n_repeats": n_repeats,
            "n_rounds": n_rounds,
            "batch_size": batch_size,
            "initial_n": initial_n,
            "strategies": strategies,
        },
        outputs={"dir": str(out_dir.relative_to(root))},
        random_seed=seed,
        gate={"passed": True},
        notes="BO-EVO-style paired multi-round retrospective campaigns on measured fitness landscape.",
    )
    return {
        "round_table": round_table,
        "selected_table": selected_table,
        "summary": summary,
        "manifest_path": manifest,
        "output_dir": out_dir,
    }


def main() -> None:
    """CLI entry point for BO-EVO-style campaign benchmarks.

    Parameters
    ----------
    None
        No command-line arguments are accepted.

    Returns
    -------
    None
        Prints campaign summary and output directory to stdout.
    """
    result = run_campaign_benchmark()
    print(result["summary"].to_string(index=False))
    print(f"Wrote: {result['output_dir']}")


if __name__ == "__main__":
    main()
