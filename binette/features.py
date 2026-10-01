"""Immutable contig evidence and bounded, ordered CheckM2 feature batches."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from checkm2 import keggData
from scipy import sparse


@dataclass(frozen=True)
class FeaturePlan:
    metadata: tuple[str, ...]
    kos: tuple[str, ...]
    groups: tuple[sparse.csc_matrix, ...]
    denominators: tuple[np.ndarray, ...]
    group_names: tuple[tuple[str, ...], ...]

    @property
    def columns(self) -> tuple[str, ...]:
        return self.metadata + self.kos + sum(self.group_names, ())


@lru_cache(maxsize=1)
def feature_plan() -> FeaturePlan:
    calculator = keggData.KeggCalculator()
    kos = tuple(calculator.return_default_values_from_category("KO_Genes"))
    ko_index = {name: index for index, name in enumerate(kos)}
    groups, denominators, group_names = [], [], []
    for group in ("KO_Pathways", "KO_Modules", "KO_Categories"):
        if group == "KO_Modules":
            names = tuple(calculator.module_definitions)
            definitions = [calculator.module_definitions[name] for name in names]
        else:
            names = tuple(calculator.return_default_values_from_category(group))
            mapping = calculator.path_category_mapping
            definitions = [
                mapping.loc[mapping[group] == name, "Kegg_ID"].values for name in names
            ]
        rows, cols = [], []
        for index, definition in enumerate(definitions):
            for ko in definition:
                if ko in ko_index:
                    rows.append(ko_index[ko])
                    cols.append(index)
        # Duplicated entries remain multiplicities, matching NumPy column selection.
        groups.append(
            sparse.csc_matrix(
                (np.ones(len(rows), dtype=np.int64), (rows, cols)),
                shape=(len(kos), len(names)),
            )
        )
        denominators.append(
            np.asarray([len(item) for item in definitions], dtype=np.int64)
        )
        group_names.append(names)
    return FeaturePlan(
        tuple(calculator.return_proper_order("Metadata")),
        kos,
        tuple(groups),
        tuple(denominators),
        tuple(group_names),
    )


def _checked_counts(values: list[int]) -> np.ndarray:
    if any(int(value) != value or value < 0 for value in values):
        raise ValueError("Contig counts must be nonnegative integers")
    if sum(map(int, values)) > np.iinfo(np.int64).max:
        raise OverflowError("Contig count reduction exceeds int64")
    dtype = np.uint32 if max(values, default=0) <= np.iinfo(np.uint32).max else np.int64
    return np.asarray(values, dtype=dtype)


@dataclass(frozen=True)
class ContigEvidence:
    ids: np.ndarray
    metadata: np.ndarray
    ko_counts: sparse.csr_matrix

    @classmethod
    def from_dicts(
        cls, kegg: dict, cds: dict, amino: dict, lengths: dict
    ) -> ContigEvidence:
        plan = feature_plan()
        ids = np.asarray(
            sorted(set(kegg) | set(cds) | set(amino) | set(lengths)), dtype=np.int64
        )
        if len(ids) and ids[0] < 0:
            raise ValueError("Contig IDs must be nonnegative")
        columns = []
        for name in plan.metadata:
            if name == "CDS":
                values = [cds.get(int(contig), 0) for contig in ids]
            elif name == "AALength":
                values = [lengths.get(int(contig), 0) for contig in ids]
            else:
                values = [amino.get(int(contig), {}).get(name, 0) for contig in ids]
            columns.append(_checked_counts(values))
        metadata = np.column_stack(columns)
        ko_index = {name: index for index, name in enumerate(plan.kos)}
        rows, cols, values = [], [], []
        for row, contig in enumerate(ids):
            for name, count in kegg.get(int(contig), {}).items():
                if name in ko_index:
                    rows.append(row)
                    cols.append(ko_index[name])
                    values.append(count)
        counts = _checked_counts(values)
        ko_counts = sparse.csr_matrix(
            (counts, (rows, cols)), shape=(len(ids), len(plan.kos))
        )
        for array in (
            ids,
            metadata,
            ko_counts.data,
            ko_counts.indices,
            ko_counts.indptr,
        ):
            array.flags.writeable = False
        return cls(ids, metadata, ko_counts)

    def save(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for name, array in (
            ("ids", self.ids),
            ("metadata", self.metadata),
            ("ko_data", self.ko_counts.data),
            ("ko_indices", self.ko_counts.indices),
            ("ko_indptr", self.ko_counts.indptr),
        ):
            np.save(root / f"{name}.npy", array, allow_pickle=False)

    @classmethod
    def load(cls, root: Path) -> ContigEvidence:
        arrays = {
            name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("ids", "metadata", "ko_data", "ko_indices", "ko_indptr")
        }
        matrix = sparse.csr_matrix(
            (arrays["ko_data"], arrays["ko_indices"], arrays["ko_indptr"]),
            shape=(len(arrays["ids"]), len(feature_plan().kos)),
            copy=False,
        )
        return cls(arrays["ids"], arrays["metadata"], matrix)

    def vectors(self, memberships: list) -> np.ndarray:
        plan = feature_plan()
        rows, cols = [], []
        for row, membership in enumerate(memberships):
            ids = np.fromiter(membership, dtype=np.int64)
            positions = np.searchsorted(self.ids, ids)
            valid = positions < len(self.ids)
            # Contigs with no genes/hits contribute zeros, including absent IDs.
            valid[valid] &= self.ids[positions[valid]] == ids[valid]
            rows.extend([row] * int(valid.sum()))
            cols.extend(positions[valid])
        incidence = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.int64), (rows, cols)),
            shape=(len(memberships), len(self.ids)),
        )
        used = np.unique(np.asarray(cols, dtype=np.intp))
        local = incidence[:, used]
        metadata = local @ self.metadata[used].astype(np.int64, copy=False)
        kos = local @ self.ko_counts[used].astype(np.int64, copy=False)
        presence = kos.copy()
        presence.data[presence.data > 1] = 1
        derived = []
        for index, (group, denominator) in enumerate(
            zip(plan.groups, plan.denominators, strict=True)
        ):
            numerator = ((kos if index == 1 else presence) @ group).toarray()
            result = np.zeros(numerator.shape, dtype=np.float64)
            np.divide(numerator, denominator, out=result, where=denominator != 0)
            derived.append(result)
        # Only the model batch is dense. Preserve baseline float64 input behavior.
        return np.concatenate((metadata, kos.toarray(), *derived), axis=1).astype(
            np.float64, copy=False
        )
