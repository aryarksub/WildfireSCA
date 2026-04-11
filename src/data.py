import os
import glob
import re
import csv
import sys
from typing import Dict, List, Optional
from functools import partial

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import rasterio
from scipy.ndimage import binary_dilation, distance_transform_edt
import numpy as np
from scipy.spatial import cKDTree


DEFAULT_REQUIRED_VARS = [
    "fire_spread/fline",
    "fire_spread/fperim",
    "fire_spread/nfp",
    "fire_spread/burned_state",
    "fire_spread/burned_state_combined",
    "fire_spread/frp",
    "fuel_topo/adj",
    "fuel_topo/cbd",
    "fuel_topo/cbh",
    "fuel_topo/cc",
    "fuel_topo/ch",
    "high_res_climate/lh",
    "high_res_climate/lw",
    "high_res_climate/m1",
    "high_res_climate/m10",
    "high_res_climate/m100",
    "high_res_climate/wd",
    "high_res_climate/ws",
    "landfire/evt",
    "landfire/fbfm13",
    "landfire/fbfm40",
    "landfire/roads",
    "landfire/asp",
    "landfire/elev",
    "landfire/slpd",
    "low_res_climate/t2m",
    "low_res_climate/d2m",
    "low_res_climate/tp",
    "low_res_climate/sp",
]

NON_CONTINUOUS_VARS = {
    "fire_spread/fline",
    "fire_spread/fperim",
    "fire_spread/nfp",
    # "landfire/evt",
    # "landfire/fbfm13",
    # "landfire/fbfm40",
    # "landfire/roads",
}


def _align_index(idx: int, T_tgt: int, T_src: int) -> int:
    if T_src <= 1:
        return 0
    if T_tgt <= 1:
        return min(idx, T_src - 1)
    return int(round(idx * (T_src - 1) / (T_tgt - 1)))


def _resize_frame(frame: torch.Tensor, size, mode: str = "bilinear") -> torch.Tensor:
    x = frame.unsqueeze(0).unsqueeze(0)
    x = F.interpolate(
        x, size=size, mode=mode, align_corners=False if mode == "bilinear" else None
    )
    return x.squeeze(0).squeeze(0)


def _is_continuous(var_key: str) -> bool:
    return var_key not in NON_CONTINUOUS_VARS


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


class GeoTiffDatasetStructured(Dataset):
    """
    PyTorch Dataset that returns a structured dictionary of all variables.
    Each GeoTIFF is treated as one variable (filename stem).
    Time is stored in GeoTIFF bands. Static rasters have a single band.

    Returns: {
        'event_name': str,
        'variables': {
            'category/var': {
                'var_name': str,
                'category': str,
                'shape': (time, height, width),
                'data': tensor,
                'files': [filename]
            },
            ...
        },
        'categories': {
            'category': { 'var': <same dict as above>, ... },
            ...
        }
    }
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.fire_events = self._discover_fire_events()
        self._fire_time_cache: Dict[str, Optional[Dict]] = {}

    def _discover_fire_events(self):
        fire_events = []
        for event_dir in sorted(os.listdir(self.base_path)):
            event_path = os.path.join(self.base_path, event_dir)
            if not os.path.isdir(event_path):
                continue
            # Only keep events that have the alignment CSV.
            if not os.path.exists(os.path.join(event_path, "fire_times.csv")):
                continue
            fire_events.append((event_dir, event_path))
        return fire_events

    def _canonical_landfire_var(self, var_name: str) -> str:
        v = var_name.lower()
        v = re.sub(r"^(ak_|hi_)", "", v)
        if re.search(r"\bevt\b", v):
            return "evt"
        if re.search(r"(fbfm13|f13_)", v):
            return "fbfm13"
        if re.search(r"(fbfm40|f40_)", v):
            return "fbfm40"
        if re.search(r"roads", v):
            return "roads"
        if re.search(r"asp", v):
            return "asp"
        if re.search(r"elev", v):
            return "elev"
        if re.search(r"slpd", v):
            return "slpd"
        return var_name

    def _load_fire_time_info(self, event_path: str) -> Dict:
        # Read fire_times.csv to determine the time alignment between FEDS
        # and other variables.
        if event_path in self._fire_time_cache:
            cached = self._fire_time_cache[event_path]
            return cached

        csv_path = os.path.join(event_path, "fire_times.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing fire_times.csv in {event_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        def _as_bool(x) -> bool:
            return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}

        feds_mask_full = [_as_bool(r.get("feds", False)) for r in rows]
        first_true = next(i for i, v in enumerate(feds_mask_full) if v)
        last_true = max(i for i, v in enumerate(feds_mask_full) if v)

        # Align to 24 hours before the first FEDS detection.
        start_idx = max(0, first_true - 24)
        # End at the final FEDS observation (inclusive).
        end_idx = last_true + 1
        feds_mask = feds_mask_full[start_idx:end_idx]
        feds_true_idx = [i for i, v in enumerate(feds_mask) if v]
        times = [r.get("time") for r in rows[start_idx:end_idx]]

        info = {
            "start_idx": start_idx,
            "length": len(feds_mask),
            "feds_mask": feds_mask,
            "feds_true_idx": feds_true_idx,
            "times": times,
        }
        self._fire_time_cache[event_path] = info
        # print(event_path, info["length"], info["feds_true_idx"], file=sys.__stdout__)
        return info

    def _slice_or_pad_dynamic(self, data: torch.Tensor, time_info: Optional[Dict]) -> torch.Tensor:
        if time_info is None or data.shape[0] <= 1:
            return data

        start_idx = int(time_info["start_idx"])
        T_out = int(time_info["length"])
        T = int(data.shape[0])
        H, W = data.shape[-2:]

        if T == T_out:
            return data

        sliced = data[start_idx : min(T, start_idx + T_out)]
        if sliced.shape[0] == T_out:
            return sliced

        if sliced.shape[0] == 0:
            return torch.zeros((T_out, H, W), dtype=data.dtype)

        # Pad missing tail by repeating the last available frame.
        pad_n = T_out - sliced.shape[0]
        tail = sliced[-1:].repeat(pad_n, 1, 1)
        return torch.cat([sliced, tail], dim=0)

    def _expand_fire_to_hourly(self, data: torch.Tensor, time_info: Optional[Dict]) -> torch.Tensor:
        if time_info is None:
            return data

        T_src, H, W = data.shape
        T_out = int(time_info["length"])
        feds_true_idx = list(time_info["feds_true_idx"])
        if T_out <= 0:
            return data
        if not feds_true_idx:
            return torch.zeros((T_out, H, W), dtype=data.dtype)

        n_obs = min(int(T_src), len(feds_true_idx))
        obs_pos = feds_true_idx[:n_obs]

        out = torch.zeros((T_out, H, W), dtype=data.dtype)
        if n_obs <= 0:
            return out

        repeats = []
        for i in range(n_obs - 1):
            repeats.append(max(0, obs_pos[i + 1] - obs_pos[i]))
        repeats.append(max(0, T_out - obs_pos[-1]))

        rep = torch.tensor(repeats, dtype=torch.long)
        valid = rep > 0
        if not torch.any(valid):
            return out

        expanded_tail = torch.repeat_interleave(data[:n_obs][valid], rep[valid], dim=0)
        start = obs_pos[int(torch.nonzero(valid, as_tuple=False)[0].item())]
        end = min(T_out, start + expanded_tail.shape[0])
        out[start:end] = expanded_tail[: end - start]
        return out

    def _apply_time_alignment(
        self, category_dir: str, data: torch.Tensor, event_time_info: Optional[Dict]
    ) -> torch.Tensor:
        if data.ndim != 3:
            return data
        if event_time_info is None:
            return data
        if category_dir == "fire_spread":
            return self._expand_fire_to_hourly(data, event_time_info)
        if data.shape[0] <= 1:
            return data
        return self._slice_or_pad_dynamic(data, event_time_info)


    def __len__(self):
        return len(self.fire_events)
    
    def _add_burned_state(self, batch):
        vars_all = batch["variables"]

        required = [
            "fire_spread/fperim",
            "fire_spread/fline",
            "fire_spread/nfp",
        ]

        if not all(k in vars_all for k in required):
            return  # Cannot build burned state

        fperim = vars_all["fire_spread/fperim"]["data"]
        fline  = vars_all["fire_spread/fline"]["data"]
        nfp    = vars_all["fire_spread/nfp"]["data"]

        T, H, W = fperim.shape
        burned = torch.zeros((T, H, W), dtype=torch.float32)
        ever_burned = torch.zeros((H, W), dtype=torch.bool)

        for t in range(T):
            perim = fperim[t]
            line  = fline[t]
            new   = nfp[t]

            active = (line > 0) | (new > 0)
            inside = (perim > 0)

            # Prevent reignition: once extinguished, cannot become active again
            # If reignition should be possible, remove this if-block
            if t > 0:
                was_extinguished = burned[t-1] == 2
                active = active & (~was_extinguished)

            # Once burned, always burned
            ever_burned |= inside | active

            # Assign states
            burned[t][~ever_burned] = 0             # never burned
            burned[t][active] = 1                   # currently burning
            burned[t][ever_burned & ~active] = 2    # burned but not active

        var_key = "fire_spread/burned_state"
        var_entry = {
            "var_name": "burned_state",
            "category": "fire_spread",
            "data": burned,
            "shape": tuple(burned.shape),
            "files": [],
        }

        batch["variables"][var_key] = var_entry
        batch["categories"].setdefault("fire_spread", {})["burned_state"] = var_entry

        # Create burned_state_combined which is 0 for unburned, 1 for burned through
        burned_combined = (burned > 0).float()

        var_key_combined = "fire_spread/burned_state_combined"
        var_entry_combined = {
            "var_name": "burned_state_combined",
            "category": "fire_spread",
            "data": burned_combined,
            "shape": tuple(burned_combined.shape),
            "files": [],
        }

        batch["variables"][var_key_combined] = var_entry_combined
        batch["categories"].setdefault("fire_spread", {})["burned_state_combined"] = var_entry_combined

    def _add_burned_state_v2(self, batch):
        vars_all = batch["variables"]

        required = [
            "fire_spread/fperim",
        ]

        if not all(k in vars_all for k in required):
            return  # Cannot build burned state

        fperim = vars_all["fire_spread/fperim"]["data"]

        T, H, W = fperim.shape
        D = torch.full((H, W), -np.inf)

        structure = np.ones((3, 3))  # Moore neighborhood

        for t in range(T-1, 0, -1):
            diff = (fperim[t] == 1) & (fperim[t - 1] == 0)

            U = binary_dilation(diff, structure=structure)

            D[U] = np.maximum(D[U], t)

        states = torch.zeros_like(fperim, dtype=torch.float32)
        states[T-1][fperim[T-1] == 1] = 2

        for t in range(T-1):
            burning = fperim[t] == 1
            states[t][burning & (t < D)] = 1
            states[t][burning & ~(t < D)] = 2
        #     print(f"t={t}, burning count={(burning).sum().item()}, state 0 count={(states[t] == 0).sum().item()}, state 1 count={(states[t] == 1).sum().item()}, state 2 count={(states[t] == 2).sum().item()}", file=sys.__stdout__)
        # print(f"t={T-1}, burning count={(fperim[T-1] == 1).sum().item()}, state 0 count={(states[T-1] == 0).sum().item()}, state 1 count={(states[T-1] == 1).sum().item()}, state 2 count={(states[T-1] == 2).sum().item()}", file=sys.__stdout__)

        var_key = "fire_spread/burned_state"
        var_entry = {
            "var_name": "burned_state",
            "category": "fire_spread",
            "data": states,
            "shape": tuple(states.shape),
            "files": [],
        }

        batch["variables"][var_key] = var_entry
        batch["categories"].setdefault("fire_spread", {})["burned_state"] = var_entry

        # Create burned_state_combined which is 0 for unburned, 1 for burned
        burned_combined = (states > 0).float()

        var_key_combined = "fire_spread/burned_state_combined"
        var_entry_combined = {
            "var_name": "burned_state_combined",
            "category": "fire_spread",
            "data": burned_combined,
            "shape": tuple(burned_combined.shape),
            "files": [],
        }

        batch["variables"][var_key_combined] = var_entry_combined
        batch["categories"].setdefault("fire_spread", {})["burned_state_combined"] = var_entry_combined

    # def _distance_to_mask(self, mask: torch.Tensor) -> torch.Tensor:
    #     inv = (~mask).cpu().numpy()
    #     dist = distance_transform_edt(inv)
    #     return torch.from_numpy(dist).float()

    def _distance_to_mask(self, mask: torch.Tensor) -> torch.Tensor:
        inv = (~mask).cpu().numpy().astype(bool)

        # Step 1: True pixels → distance to nearest False
        dist = distance_transform_edt(inv)

        # Step 2: False pixels → distance to nearest OTHER False
        false_coords = np.argwhere(~inv)

        if len(false_coords) <= 1:
            dist[~inv] = np.inf
        else:
            tree = cKDTree(false_coords)
            dists, _ = tree.query(false_coords, k=2)  # self + nearest other
            nearest_other = dists[:, 1]

            for coord, d in zip(false_coords, nearest_other):
                dist[tuple(coord)] = d

        return torch.from_numpy(dist).float()

    def _compute_tau_interval(self, S_t, S_tp, t0, t1):
        # print("Computing tau interval for t0 =", t0, "t1 =", t1, file=sys.__stdout__)
        H, W = S_t.shape
        delta_t = t1 - t0

        # Mask for burning or extinguished pixels at t0 and t1
        B_or_E_t  = (S_t  > 0)
        B_or_E_tp = (S_tp > 0)

        # Mask for extinguished pixels at t0 and t1
        E_t  = (S_t  == 2)
        E_tp = (S_tp == 2)

        # print('Counts of burning/extinguished pixels:', B_or_E_t.sum().item(), B_or_E_tp.sum().item(), E_t.sum().item(), E_tp.sum().item(), file=sys.__stdout__)

        # Estimated ignition and extinction times (initialized to inf, will be updated below)
        tau_B = torch.full((H, W), float("inf"))
        tau_E = torch.full((H, W), float("inf"))

        # Edge case: same state at start and end (no imputation needed)
        same_state = (S_t == S_tp)

        tau_B[same_state & (S_t != 0)] = t0             # already burning, ignition time is initial time t0
        tau_E[same_state & (S_t == 2)] = t0             # already extinguished, extinction time is initial time t0
        tau_B[same_state & (S_t == 0)] = float("inf")   # never ignites, ignition time is inf
        tau_E[same_state & (S_t != 2)] = float("inf")   # never extinguishes, extinction time is inf

        active = ~same_state

        # print('Active pixels:', active.sum().item(), file=sys.__stdout__)

        # NOTE: The above handles transitions U->U, B->B, E->E
        # NOTE: Remaining transitions need to be handled ... U->B, U->E (passing through B), B->E

        # Flags describing availability of reference pixels at time t0 and t1
        no_BE_t0 = not B_or_E_t.any()   # True if NO burning/extinguished pixels exist at initial time t0
        has_BE_t1 = B_or_E_tp.any()     # True if ANY burning/extinguished pixels exist at end time t1

        no_E_t0 = not E_t.any()         # True if NO extinguished pixels exist at initial time t0
        has_E_t1 = E_tp.any()           # True if ANY extinguished pixels exist at end time t1

        # print('Flags:', no_BE_t0, has_BE_t1, no_E_t0, has_E_t1, file=sys.__stdout__)

        # Distances to nearest burning/extinguished pixel at t0 and t1 (computed only if needed)
        # If there are B/E pixels at the start time, compute distance to initial fire front for ignition imputation
        if not no_BE_t0:
            d_minus = self._distance_to_mask(B_or_E_t)
        else:
            d_minus = None  # not used

        # If there are B/E pixels at the end time, compute distance to final fire front for ignition imputation
        if has_BE_t1:
            d_plus = self._distance_to_mask(B_or_E_tp)
        else:
            d_plus = None  # not used

        # If there are E pixels at the start time, compute distance to initial extinguished areas for extinction imputation
        if not no_E_t0:
            d_minus_E = self._distance_to_mask(E_t)
        else:
            d_minus_E = None

        # If there are E pixels at the end time, compute distance to final extinguished areas for extinction imputation
        if has_E_t1:
            d_plus_E = self._distance_to_mask(E_tp)
        else:
            d_plus_E = None

        # print(
        #     'Distances computed:', 
        #     (d_minus.min(), d_minus.max()) if d_minus is not None else (float('inf'), float('inf')),
        #     (d_plus.min(), d_plus.max()) if d_plus is not None else (float('inf'), float('inf')),
        #     (d_minus_E.min(), d_minus_E.max()) if d_minus_E is not None else (float('inf'), float('inf')),
        #     (d_plus_E.min(), d_plus_E.max()) if d_plus_E is not None else (float('inf'), float('inf')),
        #     file=sys.__stdout__
        # )

        # Ignition case: tau^(B)
        tau_B[(S_t > 0) & active] = t0 # Already ignited at t0

        U_to_BE = (S_t == 0) & (S_tp > 0) & active

        # print('U to BE count:', U_to_BE.sum().item(), file=sys.__stdout__)

        # If there are no burning/extinguished pixels at t0 but there are at t1, impute ignition time as t1 (latest ignition possible)
        if no_BE_t0 and has_BE_t1:
            tau_B[U_to_BE] = t1
        # Otherwise, do ignition imputation based on distance to fire front at t0 and t1
        else:
            denom = d_minus + d_plus
            denom[denom == 0] = 1e-6 # add epsilon to prevent division by zero
            # print('U to BE imputation:', delta_t, d_minus[U_to_BE], d_plus[U_to_BE], (d_minus / denom)[U_to_BE], delta_t * (d_minus / denom)[U_to_BE], file=sys.__stdout__)
            tau_B[U_to_BE] = t0 + delta_t * (d_minus / denom)[U_to_BE]

        # print(tau_B[U_to_BE], file=sys.__stdout__)

        # Extinction case: tau^(E)
        tau_E[(S_t == 2) & active] = t0 # Already extinguished at t0

        # No extinction
        no_extinguish = (S_tp != 2) & active
        tau_E[no_extinguish] = float("inf")

        # B -> E transition
        B_to_E = (S_t == 1) & (S_tp == 2) & active

        # If there are no extinguished pixels at t0 but there are at t1, impute extinction time as t1 (latest extinction possible)
        if no_E_t0 and has_E_t1:
            tau_E[B_to_E] = t1
        elif no_E_t0 and not has_E_t1:
            tau_E[active] = float("inf") # nothing extinguishes (no E pixels at t0 or t1)
        # Otherwise, do extinction imputation based on distance to extinguished areas at t0 and t1
        else:
            denom_E = d_minus_E + d_plus_E
            denom_E[denom_E == 0] = 1e-6

            tau_E[B_to_E] = t0 + delta_t * (d_minus_E / denom_E)[B_to_E]

        # U -> E transition (passing through B for sequential ignition + extinction)
        U_to_E = (S_t == 0) & (S_tp == 2) & active

        # If there are no extinguished pixels at t0 but there are at t1, impute extinction time as t1 (latest extinction possible)
        if no_E_t0 and has_E_t1:
            tau_E[U_to_E] = t1
        elif no_E_t0 and not has_E_t1:
            tau_E[active] = float("inf") # nothing extinguishes (no E pixels at t0 or t1)
        # Otherwise, do ignition imputation and then extinction imputation on top of that
        else:
            denom_E = d_minus_E + d_plus_E
            denom_E[denom_E == 0] = 1e-6

            tau_B_local = tau_B[U_to_E]
            frac_E = (d_minus_E / denom_E)[U_to_E]

            tau_E[U_to_E] = tau_B_local + (t1 - tau_B_local) * frac_E

        # Ensure ordering tau^(B) <= tau^(E)
        tau_E = torch.maximum(tau_E, tau_B)

        return tau_B, tau_E
    
    def _reconstruct_states_from_tau(self, tau_B, tau_E, t0, t1):
        delta_t = t1 - t0
        H, W = tau_B.shape

        states = torch.zeros((delta_t, H, W), dtype=torch.float32)

        for k in range(delta_t):
            t_cur = t0 + k

            U = t_cur < tau_B
            B = (tau_B <= t_cur) & (t_cur < tau_E)
            E = t_cur >= tau_E

            states[k][U] = 0
            states[k][B] = 1
            states[k][E] = 2

        return states
    
    def _add_burned_state_imputed(self, batch):
        if "fire_spread/burned_state" not in batch["variables"]:
            return
        # print("Adding imputed burned state variable...", file=sys.__stdout__)

        burned = batch["variables"]["fire_spread/burned_state"]["data"]
        feds_mask = batch["time_info"]["feds_mask"]

        T, H, W = burned.shape
        # TODO: Add 0 to obs_idx for blank slate of unburned pixels to start?
        obs_idx = [i for i, v in enumerate(feds_mask) if v or i == 0]

        # print(obs_idx, file=sys.__stdout__)

        if len(obs_idx) < 2:
            return

        imputed = burned.clone()

        for i in range(len(obs_idx) - 1):
            t0 = obs_idx[i]
            t1 = obs_idx[i + 1]

            if t1 - t0 <= 1:
                continue

            S_t  = burned[t0]
            S_tp = burned[t1]

            tau_B, tau_E = self._compute_tau_interval(S_t, S_tp, t0, t1)

            ############################################################
            # finite_B = tau_B[torch.isfinite(tau_B)]
            # finite_E = tau_E[torch.isfinite(tau_E)]

            # if finite_B.numel() > 0:
            #     min_B = finite_B.min()
            #     max_B = finite_B.max()
            #     count_B_between = ((finite_B > min_B) & (finite_B < max_B)).sum().item()
            # else:
            #     min_B = max_B = float('inf')
            #     count_B_between = 0

            # if finite_E.numel() > 0:
            #     min_E = finite_E.min()
            #     max_E = finite_E.max()
            #     count_E_between = ((finite_E > min_E) & (finite_E < max_E)).sum().item()
            # else:
            #     min_E = max_E = float('inf')
            #     count_E_between = 0

            # print(
            #     t0, t1,
            #     'B:', min_B, max_B, 'count between:', count_B_between,
            #     'E:', min_E, max_E, 'count between:', count_E_between,
            #     file=sys.__stdout__
            # )
            #############################################################

            recon = self._reconstruct_states_from_tau(tau_B, tau_E, t0, t1)

            imputed[t0:t1] = recon
            imputed[t0] = burned[t0]
            imputed[t1] = burned[t1]

        var_key = "fire_spread/burned_state_imp1"
        var_entry = {
            "var_name": "burned_state_imp1",
            "category": "fire_spread",
            "data": imputed,
            "shape": tuple(imputed.shape),
            "files": [],
        }

        batch["variables"][var_key] = var_entry
        batch["categories"].setdefault("fire_spread", {})["burned_state_imp1"] = var_entry

    def __getitem__(self, idx):
        event_name, event_path = self.fire_events[idx]
        # print("******* EVENT NAME:", event_name, file=sys.__stdout__)
        batch = {"event_name": event_name, "variables": {}, "categories": {}}
        event_time_info = self._load_fire_time_info(event_path)
        batch["time_info"] = {
            "length": event_time_info["length"],
            "start_idx": event_time_info["start_idx"],
            "times": event_time_info["times"],
            "feds_mask": event_time_info["feds_mask"],
        }


        for category_dir in sorted(os.listdir(event_path)):
            category_path = os.path.join(event_path, category_dir)
            if not os.path.isdir(category_path):
                continue

            tif_files = sorted(glob.glob(os.path.join(category_path, "*.tif")))
            if not tif_files:
                continue

            for tif_file in tif_files:
                file_name = os.path.basename(tif_file)
                var_name = os.path.splitext(file_name)[0]
                if category_dir == "landfire":
                    var_name = self._canonical_landfire_var(var_name)
                try:
                    with rasterio.open(tif_file, "r") as src:
                        data = torch.from_numpy(src.read()).float()  # (time, h, w)
                except Exception as e:
                    print(f"Error loading {tif_file}: {e}")
                    continue

                data = self._apply_time_alignment(category_dir, data, event_time_info)

                var_key = f"{category_dir}/{var_name}"
                if var_key in batch["variables"]:
                    batch["variables"][var_key]["files"].append(file_name)
                    batch["categories"][category_dir][var_name]["files"].append(file_name)
                    continue

                var_entry = {
                    "var_name": var_name,
                    "category": category_dir,
                    "data": data,
                    "shape": tuple(data.shape),
                    "files": [file_name],
                }

                batch["variables"][var_key] = var_entry
                batch["categories"].setdefault(category_dir, {})[var_name] = var_entry

        # self._add_burned_state(batch)
        self._add_burned_state_v2(batch)
        self._add_burned_state_imputed(batch)
        # for vk in batch["variables"]:
        #     print(vk, batch["variables"][vk]["shape"], file=sys.__stdout__)
        #     if vk == 'fire_spread/burned_state_imp1' and batch["variables"][vk]["shape"][0] > 1:
        #         for i in range(1, batch["variables"][vk]["shape"][0]):
        #             if not torch.all(batch["variables"][vk]["data"][i-1] == batch["variables"][vk]["data"][i]).item():
        #                 print(
        #                     (
        #                         i, 
        #                         torch.all(batch["variables"][vk]["data"][i-1] == batch["variables"][vk]["data"][i]).item(),
        #                         # torch.bincount(batch["variables"][vk]["data"][i-1].flatten().to(torch.int64), minlength=3),
        #                         torch.bincount(batch["variables"][vk]["data"][i].flatten().to(torch.int64), minlength=3),
        #                         batch["variables"][vk]["data"][i-1].sum().item(),
        #                         batch["variables"][vk]["data"][i].sum().item(),
        #                         i % 12 != 0 and not torch.all(batch["variables"][vk]["data"][i-1] == batch["variables"][vk]["data"][i]).item()
        #                     ),
        #                     # torch.all(batch["variables"][vk]["data"][0] == batch["variables"][vk]["data"][1]).item(), 
        #                     # torch.all(batch["variables"][vk]["data"][1] == batch["variables"][vk]["data"][2]).item(),
        #                     # torch.all(batch["variables"][vk]["data"][2] == batch["variables"][vk]["data"][3]).item(),
        #                     # torch.all(batch["variables"][vk]["data"][0] == batch["variables"][vk]["data"][11]).item(),
        #                     # torch.all(batch["variables"][vk]["data"][0] == batch["variables"][vk]["data"][12]).item(),
        #                     # torch.all(batch["variables"][vk]["data"][12] == batch["variables"][vk]["data"][13]).item(),
        #                     # torch.all(batch["variables"][vk]["data"][12] == batch["variables"][vk]["data"][23]).item(),
        #                     # torch.all(batch["variables"][vk]["data"][0] == batch["variables"][vk]["data"][67]).item(),
        #                     file=sys.__stdout__
        #                 )
        return batch


def firecube_collate(batch):
    """
    Collate FireCube samples without stacking tensors of different shapes.
    For batch_size=1, return the single sample for convenience.
    """
    if len(batch) == 1:
        return batch[0]
    return {
        "event_name": [b["event_name"] for b in batch],
        "variables": [b["variables"] for b in batch],
        "categories": [b["categories"] for b in batch],
        "time_info": [b["time_info"] for b in batch],
    }


class OneStepDatasetSimple(Dataset):
    def __init__(
        self,
        base_dataset: GeoTiffDatasetStructured,
        required_vars: List[str],
        target_var: str = "fire_spread/nfp",
        step_hours: int = 12,
        horizon_hours: int = 12,
        hourly_agg: str = "concat",
        missing_value: float = -1.0,
        stats: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        if not required_vars:
            raise ValueError("required_vars must be a non-empty list of canonical variables")
        self.base = base_dataset
        self.required_vars = required_vars
        self.target_var = target_var
        self.step = step_hours
        self.horizon = horizon_hours
        self.hourly_agg = hourly_agg
        self.missing_value = missing_value
        self.stats = stats or {}

        self.fire_vars = [v for v in required_vars if v.split("/")[0] in ("fire_spread",)]
        self.hourly_vars = [v for v in required_vars if v.split("/")[0] in ("high_res_climate", "low_res_climate")]
        self.static_vars = [v for v in required_vars if v.split("/")[0] in ("fuel_topo", "landfire")]
        self.sample_index = self._build_sample_index()

    def _hourly_mode(self) -> str:
        mode = "concat" if self.hourly_agg is None else str(self.hourly_agg).strip().lower()
        if mode == "concat":
            return "concat"
        if mode == "mean":
            return "mean"
        raise ValueError(
            f"Unsupported hourly_agg={self.hourly_agg!r}. "
            "Use one of: mean or concat."
        )

    def _canonical_landfire_var(self, var_name: str) -> str:
        v = var_name.lower()
        v = re.sub(r"^(ak_|hi_)", "", v)
        if re.search(r"\bevt\b", v):
            return "evt"
        if re.search(r"(fbfm13|f13_)", v):
            return "fbfm13"
        if re.search(r"(fbfm40|f40_)", v):
            return "fbfm40"
        if re.search(r"roads", v):
            return "roads"
        if re.search(r"asp", v):
            return "asp"
        if re.search(r"elev", v):
            return "elev"
        if re.search(r"slpd", v):
            return "slpd"
        return var_name

    def _normalize_vars(self, vars_all: Dict) -> Dict:
        norm = {}
        for key, info in vars_all.items():
            if not key.startswith("landfire/"):
                if key not in norm:
                    norm[key] = info
                continue
            cat, var = key.split("/", 1)
            canon = self._canonical_landfire_var(var)
            canon_key = f"{cat}/{canon}"
            if canon_key not in norm:
                entry = info.copy()
                entry["var_name"] = canon
                norm[canon_key] = entry
            else:
                if "files" in norm[canon_key] and "files" in info:
                    norm[canon_key]["files"] = norm[canon_key]["files"] + info["files"]
        return norm

    def _target_timesteps_for_event(self, event_path: str) -> int:
        category, var = self.target_var.split("/", 1)
        # Use fperim to determine timesteps since there is no file for burned_state (it is derived on the fly)
        if "burned_state" in var:
            var = "fperim" 
        category_path = os.path.join(event_path, category)
        if not os.path.isdir(category_path):
            return 0
        for tif_file in sorted(glob.glob(os.path.join(category_path, "*.tif"))):
            file_name = os.path.splitext(os.path.basename(tif_file))[0]
            cand = file_name
            if category == "landfire":
                cand = self._canonical_landfire_var(cand)
            if cand != var:
                continue
            try:
                with rasterio.open(tif_file, "r") as src:
                    return int(src.count)
            except Exception:
                return 0
        return 0

    def _candidate_ts(self, T_tgt: int) -> List[int]:
        if T_tgt <= 0:
            return [0]
        candidates = list(range(self.step, max(self.step, T_tgt - self.horizon), self.step))
        if not candidates:
            return [max(0, T_tgt - self.horizon - 1)]
        return candidates

    def _build_sample_index(self):
        index = []
        for event_idx, (_, event_path) in enumerate(self.base.fire_events):
            ti = self.base._load_fire_time_info(event_path)
            T = int(ti["length"])
            mask = ti["feds_mask"]
            # For imputation version
            if self.step < 12:
                for t in range(self.horizon, T - self.horizon, self.step):
                    index.append((event_idx, t))
            # For regular version (12-hour timestep with 12-hour horizon)
            else:
                for t in range(self.step, T - self.horizon, self.step):
                    if mask[t + self.horizon]:
                        index.append((event_idx, t))
        return index


    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        event_idx, t = self.sample_index[idx]
        sample = self.base[event_idx]
        vars_all = self._normalize_vars(sample["variables"])
        hourly_mode = self._hourly_mode()

        tgt = vars_all[self.target_var]["data"]  # (T, H, W)
        T_tgt, H_tgt, W_tgt = tgt.shape
        target_size = (H_tgt, W_tgt)

        fire_frames = []
        for k in self.fire_vars:
            if k in vars_all:
                data = vars_all[k]["data"]
                t_src = _align_index(t, T_tgt, data.shape[0])
                frame = data[t_src]
                frame = _resize_frame(frame, target_size, mode="bilinear")
                frame = torch.nan_to_num(frame, nan=self.missing_value)
                frame = torch.where(frame <= -9990.0, torch.full_like(frame, self.missing_value), frame)
                if _is_continuous(k) and k in self.stats:
                    mn = self.stats[k]["min"]
                    mx = self.stats[k]["max"]
                    if mx > mn:
                        frame = (frame - mn) / (mx - mn)
                        frame = torch.clamp(frame, 0.0, 1.0)
                fire_frames.append(frame)
            else:
                fire_frames.append(torch.full(target_size, self.missing_value))
        x_fire = torch.stack(fire_frames, dim=0)

        hourly_frames = []
        time_window = list(range(max(0, t - self.step + 1), t + 1))
        if len(time_window) < self.step:
            pad_val = time_window[0] if time_window else 0
            time_window = [pad_val] * (self.step - len(time_window)) + time_window
        for k in self.hourly_vars:
            if k not in vars_all:
                if hourly_mode == "mean":
                    frames = torch.full(target_size, self.missing_value)
                elif hourly_mode == "concat":
                    frames = torch.full((len(time_window), *target_size), self.missing_value)
                else:
                    raise ValueError("Unsupported hourly_agg")
                hourly_frames.append(frames)
                continue
            data = vars_all[k]["data"]
            frames = []
            for ti in time_window:
                ti_src = _align_index(ti, T_tgt, data.shape[0])
                frame = data[ti_src]
                frame = _resize_frame(frame, target_size, mode="bilinear")
                frame = torch.nan_to_num(frame, nan=self.missing_value)
                frame = torch.where(frame <= -9990.0, torch.full_like(frame, self.missing_value), frame)
                if _is_continuous(k) and k in self.stats:
                    mn = self.stats[k]["min"]
                    mx = self.stats[k]["max"]
                    if mx > mn:
                        frame = (frame - mn) / (mx - mn)
                        frame = torch.clamp(frame, 0.0, 1.0)
                frames.append(frame)
            frames = torch.stack(frames, dim=0)
            if hourly_mode == "mean":
                frames = frames.mean(dim=0)
            elif hourly_mode != "concat":
                raise ValueError("Unsupported hourly_agg")
            hourly_frames.append(frames)
        x_hourly = torch.stack(hourly_frames, dim=0)

        static_frames = []
        for k in self.static_vars:
            if k not in vars_all:
                static_frames.append(torch.full(target_size, self.missing_value))
                continue
            data = vars_all[k]["data"]
            frame = data[0]
            frame = _resize_frame(frame, target_size, mode="bilinear")
            frame = torch.nan_to_num(frame, nan=0.0)
            frame = torch.where(frame <= -9990.0, torch.zeros_like(frame), frame)

            if _is_continuous(k) and k in self.stats:
                mn = self.stats[k]["min"]
                mx = self.stats[k]["max"]
                if mx > mn:
                    frame = (frame - mn) / (mx - mn)
                    frame = torch.clamp(frame, 0.0, 1.0)
            static_frames.append(frame)
        x_static = torch.stack(static_frames, dim=0)

        # Imputation version
        if self.step < 12:
            t_y = min(t + self.step, T_tgt - 1)
        # Non-imputation version
        else:
            t_y = min(t + self.horizon, T_tgt - 1)
        y = tgt[t_y].unsqueeze(0)
        if self.target_var.split("/")[0] in ("fire_spread"):
            y = torch.nan_to_num(y, nan=0.0)
            y = torch.where(y <= -9990.0, torch.zeros_like(y), y)

        x_hourly_flat = x_hourly.reshape(-1, *target_size)
        x_all = torch.cat([x_fire, x_hourly_flat, x_static], dim=0)

        return {
            "event_name": sample["event_name"],
            "t": t,
            "x_fire": x_fire,
            "x_hourly": x_hourly,
            "x_static": x_static,
            "x_all": x_all,
            "y": y,
            "fire_vars": self.fire_vars,
            "hourly_vars": self.hourly_vars,
            "static_vars": self.static_vars,
        }


def _pad_spatial(t: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    if t.shape[-2] == target_h and t.shape[-1] == target_w:
        return t
    pad_h = target_h - t.shape[-2]
    pad_w = target_w - t.shape[-1]
    return F.pad(t, (0, pad_w, 0, pad_h))


def _onestep_collate_impl(batch, out_type: str = "structured"):
    max_h = max(b["y"].shape[-2] for b in batch)
    max_w = max(b["y"].shape[-1] for b in batch)

    out = {
        "event_name": [b["event_name"] for b in batch],
        "t": [b["t"] for b in batch],
        "fire_vars": batch[0]["fire_vars"],
        "hourly_vars": batch[0]["hourly_vars"],
        "static_vars": batch[0]["static_vars"],
    }

    y_list = []
    masks = []
    for b in batch:
        y = _pad_spatial(b["y"], max_h, max_w)
        y_list.append(y)
        m = torch.zeros((1, max_h, max_w), dtype=torch.bool)
        m[:, : b["y"].shape[-2], : b["y"].shape[-1]] = True
        masks.append(m)
    out["y"] = torch.stack(y_list, dim=0)
    out["mask"] = torch.stack(masks, dim=0)

    if out_type == "flattened":
        xs = [b.get("x", b["x_all"]) for b in batch]
        xs = [_pad_spatial(x, max_h, max_w) for x in xs]
        out["x"] = torch.stack(xs, dim=0)
        return out

    def _stack(key: str):
        xs = [_pad_spatial(b[key], max_h, max_w) for b in batch]
        return torch.stack(xs, dim=0)

    out["x_fire"] = _stack("x_fire")
    out["x_hourly"] = _stack("x_hourly")
    out["x_static"] = _stack("x_static")
    out["x_all"] = _stack("x_all")
    return out


def make_onestep_collate(out_type: str = "structured"):
    # Return a top-level callable so it is picklable under multiprocessing spawn (DDP).
    return partial(_onestep_collate_impl, out_type=out_type)


def compute_global_min_max(
    base_dataset: GeoTiffDatasetStructured,
    required_vars: List[str],
    sample_limit: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    idxs = range(len(base_dataset))
    if sample_limit is not None:
        idxs = list(idxs)[:sample_limit]

    for i in idxs:
        sample = base_dataset[i]
        vars_all = sample["variables"]
        for k in required_vars:
            if not _is_continuous(k):
                continue
            if k not in vars_all:
                continue
            data = vars_all[k]["data"]
            # Ignore NaNs and sentinel nodata values when computing stats.
            data = data.float()
            valid = torch.isfinite(data) & (data > -9990.0)
            if not torch.any(valid):
                continue
            vmin = float(torch.min(data[valid]).item())
            vmax = float(torch.max(data[valid]).item())
            if k not in stats:
                stats[k] = {"min": vmin, "max": vmax}
            else:
                stats[k]["min"] = min(stats[k]["min"], vmin)
                stats[k]["max"] = max(stats[k]["max"], vmax)
    return stats


def build_onestep_loader(
    base_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    out_type: str = "structured",
    required_vars: List[str] = None,
    target_var: str = "fire_spread/nfp",
    step_hours: int = 12,
    horizon_hours: int = 12,
    hourly_agg: str = "concat",
    missing_value: float = -1.0,
    compute_stats: bool = True,
    stats_path: Optional[str] = None,
    stats_sample_limit: Optional[int] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    required_vars = required_vars or DEFAULT_REQUIRED_VARS
    base = GeoTiffDatasetStructured(base_path)
    stats = {}
    use_dist_sync = distributed and world_size > 1 and _dist_ready()
    if use_dist_sync:
        obj = [None]
        if rank == 0:
            if compute_stats:
                stats = compute_global_min_max(
                    base, required_vars, sample_limit=stats_sample_limit
                )
                if stats_path:
                    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
                    import json

                    with open(stats_path, "w", encoding="utf-8") as f:
                        json.dump(stats, f, indent=2)
            elif stats_path and os.path.exists(stats_path):
                import json

                with open(stats_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)
            obj[0] = stats
        dist.broadcast_object_list(obj, src=0)
        stats = obj[0] or {}
    else:
        if compute_stats:
            stats = compute_global_min_max(
                base, required_vars, sample_limit=stats_sample_limit
            )
            if stats_path:
                os.makedirs(os.path.dirname(stats_path), exist_ok=True)
                import json

                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
        elif stats_path and os.path.exists(stats_path):
            import json

            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
    onestep_dataset = OneStepDatasetSimple(
        base,
        required_vars=required_vars,
        target_var=target_var,
        step_hours=step_hours,
        horizon_hours=horizon_hours,
        hourly_agg=hourly_agg,
        missing_value=missing_value,
        stats=stats,
    )
    loader = DataLoader(
        onestep_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_onestep_collate(out_type),
    )
    return onestep_dataset, loader
