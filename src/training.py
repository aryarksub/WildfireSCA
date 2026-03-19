import random
import sys
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset
import yaml

from data import build_onestep_loader
from models import MLPSCA, DirectLogisticSCA

FORCED_TEST_EVENT_IDS = []


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ============================================================
# Train / Val / Test Split
# ============================================================

def _dataset_event_name(dataset, idx):
    # Frame-level dataset: index maps into dataset.sample_index -> event index.
    if hasattr(dataset, "sample_index") and hasattr(dataset, "base") and hasattr(dataset.base, "fire_events"):
        event_idx, _t = dataset.sample_index[idx]
        return dataset.base.fire_events[event_idx][0]
    # OneStepDatasetSimple wraps GeoTiffDatasetStructured in `base`.
    if hasattr(dataset, "base") and hasattr(dataset.base, "fire_events"):
        return dataset.base.fire_events[idx][0]
    # GeoTiffDatasetStructured exposes fire_events directly.
    if hasattr(dataset, "fire_events"):
        return dataset.fire_events[idx][0]
    # Fallback (slower): load sample.
    return dataset[idx]["event_name"]

def hardcoded_split(dataset, forced_test_ids=None):
    # Train/val/test split (hardcoded seed + optional forced test event IDs)
    split_seed = 123
    val_frac = 0.2
    test_frac = 0.2
    train_frac = 1 - val_frac - test_frac
    if forced_test_ids is None:
        forced = set(FORCED_TEST_EVENT_IDS)
    elif isinstance(forced_test_ids, str):
        forced = {forced_test_ids}
    else:
        forced = set(forced_test_ids)

    n = len(dataset)
    indices = list(range(n))
    event_by_idx = {i: _dataset_event_name(dataset, i) for i in indices}

    test_idx = [i for i in indices if event_by_idx[i] in forced]
    remaining = [i for i in indices if event_by_idx[i] not in forced]

    remaining_events = sorted({event_by_idx[i] for i in remaining})

    rng = random.Random(split_seed)
    rng.shuffle(remaining_events)

    n_events = len(remaining_events)

    n_train = int(train_frac * n_events)
    n_val   = int(val_frac * n_events)

    train_events = set(remaining_events[:n_train])
    val_events   = set(remaining_events[n_train:n_train+n_val])
    test_events  = set(remaining_events[n_train+n_val:])

    # ----------------------------------
    # Convert event splits -> indices
    # ----------------------------------

    train_idx = [i for i in indices if event_by_idx[i] in train_events]
    val_idx   = [i for i in indices if event_by_idx[i] in val_events]
    test_idx  = [i for i in indices if event_by_idx[i] in test_events]

    return train_idx, val_idx, test_idx


def get_training_objects(
    data_cfg,
    device,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    model_backbone: str = 'logistic',
    num_states: int = 3,
):
    dataset, loader = build_onestep_loader(
        base_path=data_cfg["base_path"],
        batch_size=data_cfg.get("batch_size", 8),
        shuffle=bool(data_cfg.get("shuffle", True)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        out_type="structured", # default for train and val
        required_vars=data_cfg.get("required_vars"),
        target_var=data_cfg.get("target_var", "fire_spread/burned_state"),
        step_hours=int(data_cfg.get("step_hours", 12)),
        horizon_hours=int(data_cfg.get("horizon_hours", 12)),
        hourly_agg=data_cfg.get("hourly_agg", "concat"),
        missing_value=float(data_cfg.get("missing_value", -1.0)),
        compute_stats=bool(data_cfg.get("compute_stats", False)),
        stats_path=data_cfg.get("stats_path", None),
        stats_sample_limit=data_cfg.get("stats_sample_limit"),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    example = dataset[0]
    # print(example.keys(), file=sys.__stdout__)
    # print(len(dataset), example['x_fire'].sum().item(), example['y'].sum().item(), file=sys.__stdout__)
    n_covariates = example["x_all"].shape[0] - 1
    # print('Number of x fire:', example['x_fire'].shape, file=sys.__stdout__)
    # print('Number of x hourly:', example['x_hourly'].shape, file=sys.__stdout__)
    # print('Number of x static:', example['x_static'].shape, file=sys.__stdout__)
    # print('Number of covariates:', n_covariates, example['x_all'].shape, file=sys.__stdout__)
    # print('y:', example['y'].shape, file=sys.__stdout__)
    # print('fire vars:', example['fire_vars'], file=sys.__stdout__)
    # print('hourly vars:', example['hourly_vars'], file=sys.__stdout__)
    # print('static vars:', example['static_vars'], file=sys.__stdout__)

    train_idx, val_idx, _test_idx = hardcoded_split(dataset)
    # print(_test_idx, file=sys.__stdout__)

    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(data_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = data_cfg.get("prefetch_factor", 2)
    loader_kwargs = {
        "batch_size": data_cfg.get("batch_size", 8),
        "num_workers": num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, _test_idx)
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_sampler = DistributedSampler(
            test_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        train_loader = DataLoader(
            train_ds,
            shuffle=False,
            sampler=train_sampler,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            sampler=val_sampler,
            **loader_kwargs,
        )
        test_loader = DataLoader(
            test_ds,
            shuffle=False,
            sampler=test_sampler,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            **loader_kwargs,
        )
        test_loader = DataLoader(
            test_ds,
            shuffle=False,
            **loader_kwargs,
        )

    if model_backbone == 'logistic':
        model = DirectLogisticSCA(n_covariates, num_states=num_states).to(device)
    elif model_backbone == 'mlp':
        model = MLPSCA(n_covariates, num_states=num_states).to(device)
    else:
        raise ValueError(f"Unsupported model_backbone={model_backbone}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return train_loader, val_loader, test_loader, model, optimizer
