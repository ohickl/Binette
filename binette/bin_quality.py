#!/usr/bin/env python3
import logging
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from itertools import chain, islice

import numpy as np
import pandas as pd
from checkm2 import keggData
from rich.progress import Progress

from binette.bin_manager import Bin

logger = logging.getLogger(__name__)

# Suppress unnecessary TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.FATAL)

# Lazy loaders for checkm2 components that import keras
# These will only be imported when explicitly called
_modelPostprocessing = None
_modelProcessing = None
_keras_initialized = False


def _initialize_keras_environment():
    """Initialize TensorFlow/Keras to ensure thread safety and memory management"""
    global _keras_initialized
    if not _keras_initialized:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings

        try:
            # Only import keras-related modules when needed
            import tensorflow as tf

            # Use a single thread for predictions to avoid thread contention
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)

            _keras_initialized = True
            logger.debug("TensorFlow/Keras environment initialized")
        except Exception as e:
            logger.warning(f"Failed to fully initialize TensorFlow environment: {e!s}")


def get_modelPostprocessing():
    """Lazy load modelPostprocessing module only when needed"""
    global _modelPostprocessing
    if _modelPostprocessing is None:
        # Initialize Keras environment
        _initialize_keras_environment()

        # Only import keras when absolutely needed
        from checkm2 import modelPostprocessing

        _modelPostprocessing = modelPostprocessing
    return _modelPostprocessing


def get_modelProcessing():
    """Lazy load modelProcessing module only when needed"""
    global _modelProcessing
    if _modelProcessing is None:
        # Initialize Keras environment
        _initialize_keras_environment()

        # Only import keras when absolutely needed
        from checkm2 import modelProcessing

        _modelProcessing = modelProcessing

    return _modelProcessing


def get_bins_metadata_df(
    bins: list[Bin],
    contig_to_cds_count: dict[str, int],
    contig_to_aa_counter: dict[str, Counter],
    contig_to_aa_length: dict[str, int],
) -> pd.DataFrame:
    """
    Optimized: Generate a DataFrame containing metadata for a list of bins.
    Handles contigs that appear in multiple bins.
    """
    metadata_order = keggData.KeggCalculator().return_proper_order("Metadata")
    bin_keys = [b.contigs_key for b in bins]

    # --- Pre-aggregate CDS and AA length ---
    cds_per_bin = defaultdict(int)
    aa_len_per_bin = defaultdict(int)
    aa_counter_per_bin = defaultdict(Counter)

    # map contigs → all bins they belong to
    contig_to_bins = defaultdict(list)
    for b in bins:
        for c in b.contigs:
            contig_to_bins[c].append(b.contigs_key)

    # Only contigs in this batch contribute; unrelated assembly evidence is not scanned.
    for contig in contig_to_bins:
        cds = contig_to_cds_count.get(contig, 0)
        for bin_key in contig_to_bins.get(contig, []):
            cds_per_bin[bin_key] += cds

    # distribute AA lengths
    for contig in contig_to_bins:
        length = contig_to_aa_length.get(contig, 0)
        for bin_key in contig_to_bins.get(contig, []):
            aa_len_per_bin[bin_key] += length

    # distribute AA counters
    for contig in contig_to_bins:
        counter = contig_to_aa_counter.get(contig, {})
        for bin_key in contig_to_bins.get(contig, []):
            aa_counter_per_bin[bin_key].update(counter)

    # --- Build rows ---
    rows = []
    for key in bin_keys:
        row = {
            "Name": key,
            "CDS": cds_per_bin.get(key, 0),
            "AALength": aa_len_per_bin.get(key, 0),
        }
        row.update(aa_counter_per_bin.get(key, {}))
        rows.append(row)

    # --- Construct DataFrame directly ---
    metadata_df = pd.DataFrame(rows).fillna(0)

    # Ensure column order
    all_cols = ["Name"] + metadata_order
    for col in metadata_order:
        if col not in metadata_df.columns:
            metadata_df[col] = 0

    metadata_df = metadata_df[all_cols].astype(dict.fromkeys(metadata_order, int))
    metadata_df = metadata_df.set_index("Name", drop=False)

    return metadata_df


def get_diamond_feature_per_bin_df(
    bins: list[Bin], contig_to_kegg_counter: dict[str, Counter]
) -> tuple[pd.DataFrame, int]:
    """
    Optimized: Generate a DataFrame containing Diamond feature counts per bin,
    including KEGG KO counts and completeness information for pathways, categories, and modules.
    Handles contigs that may belong to multiple bins.
    """
    KeggCalc = keggData.KeggCalculator()
    defaultKOs = KeggCalc.return_default_values_from_category("KO_Genes")
    bin_keys = [b.contigs_key for b in bins]

    # --- Aggregate KO counters per bin ---
    bin_to_ko_counter = {}
    for bin_obj in bins:
        bin_ko_counter = Counter()
        for contig in bin_obj.contigs:
            ko_counter = contig_to_kegg_counter.get(contig)
            if ko_counter:
                bin_ko_counter.update(ko_counter)
        bin_to_ko_counter[bin_obj.contigs_key] = bin_ko_counter

    # --- Build KO count DataFrame directly ---
    ko_count_per_bin_df = (
        pd.DataFrame.from_dict(bin_to_ko_counter, orient="index")
        .reindex(bin_keys)  # keep bin order
        .fillna(0)
        .astype(int)
    )

    # Ensure all defaultKOs exist
    ko_count_per_bin_df = ko_count_per_bin_df.reindex(
        columns=list(defaultKOs), fill_value=0
    )

    # ko_count_per_bin_df.index.name = "Name"
    ko_count_per_bin_df["Name"] = ko_count_per_bin_df.index

    # --- Calculate higher-level completeness ---
    logger.debug("Calculating completeness of pathways, categories, and modules")
    KO_pathways = calculate_KO_group(KeggCalc, "KO_Pathways", ko_count_per_bin_df)
    KO_categories = calculate_KO_group(KeggCalc, "KO_Categories", ko_count_per_bin_df)

    KO_modules = calculate_module_completeness(KeggCalc, ko_count_per_bin_df)

    # --- Concatenate results ---
    diamond_complete_results = pd.concat(
        [ko_count_per_bin_df, KO_pathways, KO_modules, KO_categories], axis=1
    )

    return diamond_complete_results, len(defaultKOs)


def calculate_KO_group(
    KeggCalc: keggData.KeggCalculator, group: str, KO_gene_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the completeness of KEGG feature groups per bin.

    :param KeggCalc: An instance of KeggCalculator containing KEGG mappings.
    :param group: Feature group name (e.g., "KO_Pathways", "KO_Categories").
    :param KO_gene_data: DataFrame containing KO counts per bin with last column "Name".

    :return: DataFrame with completeness values for each feature vector in the group.
    """

    # last column is 'Name'
    data = KO_gene_data.drop(columns=["Name"]).values
    n_bins = data.shape[0]

    # Build output DataFrame
    ordered_entries = KeggCalc.return_default_values_from_category(group)
    feature_vectors = list(ordered_entries.keys())
    n_features = len(feature_vectors)

    # Create empty numpy array for results
    result = np.zeros((n_bins, n_features), dtype=float)

    # Map Kegg_IDs to column indices in KO_gene_data
    col_map = {ko: idx for idx, ko in enumerate(KO_gene_data.columns[:-1])}

    for f_idx, vector in enumerate(feature_vectors):
        # KOs belonging to this feature vector
        kegg_ids = KeggCalc.path_category_mapping.loc[
            KeggCalc.path_category_mapping[group] == vector, "Kegg_ID"
        ].values

        # Only keep KOs present in DataFrame columns
        present_cols = [col_map[ko] for ko in kegg_ids if ko in col_map]
        if not present_cols:
            continue

        # Presence/absence: values >1 -> 1
        vals = data[:, present_cols]
        vals[vals > 1] = 1
        result[:, f_idx] = vals.sum(axis=1) / len(kegg_ids)

    return pd.DataFrame(result, columns=feature_vectors, index=KO_gene_data.index)


def calculate_module_completeness(
    KeggCalc: keggData.KeggCalculator, KO_gene_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute module completeness per bin using NumPy for speed.

    :param KeggCalc: An instance of KeggCalculator containing module definitions.
    :param KO_gene_data: DataFrame containing KO counts per bin with last column "Name".

    :return: DataFrame with completeness values for each module.
    """
    data = KO_gene_data.drop(columns=["Name"]).values
    n_bins = data.shape[0]

    modules = list(KeggCalc.module_definitions.keys())
    n_modules = len(modules)

    # Map KO names to column indices
    col_map = {
        ko: idx for idx, ko in enumerate(KO_gene_data.drop(columns=["Name"]).columns)
    }

    # Prepare result array
    result = np.zeros((n_bins, n_modules), dtype=float)

    for m_idx, module in enumerate(modules):
        # Only keep KOs that exist in the DataFrame

        module_kos = [ko for ko in KeggCalc.module_definitions[module] if ko in col_map]
        if not module_kos:
            continue
        cols = [col_map[ko] for ko in module_kos]

        vals = data[:, cols]
        # vals[vals > 1] = 1  # presence/absence

        result[:, m_idx] = vals.sum(axis=1) / len(KeggCalc.module_definitions[module])

    return pd.DataFrame(result, columns=modules, index=KO_gene_data.index)


def prepare_contig_sizes(contig_to_size: dict[int, int]) -> np.ndarray:
    """
    Prepare a numpy array of contig sizes for fast access.

    :param contig_to_size: Dictionary mapping contig IDs to contig sizes.

    :return: Numpy array where the index corresponds to the contig ID
             and the value is the contig size.
    """
    if not contig_to_size:
        return np.zeros(0, dtype=np.int64)
    if any(not isinstance(key, int) or key < 0 for key in contig_to_size) or any(
        int(value) != value or value < 0 for value in contig_to_size.values()
    ):
        raise ValueError("Contig IDs and sizes must be nonnegative integers")
    if sum(map(int, contig_to_size.values())) > np.iinfo(np.int64).max:
        raise OverflowError("Assembly length exceeds int64 reduction")
    max_id = max(contig_to_size)
    contig_sizes = np.zeros(max_id + 1, dtype=np.int64)
    for contig_id, size in contig_to_size.items():
        contig_sizes[contig_id] = size
    return contig_sizes


def compute_N50(lengths: np.ndarray) -> int:
    """
    Compute the N50 value for a given set of contig lengths.

    :param lengths: Numpy array of contig lengths.

    :return: N50 value (contig length at which 50% of the genome is covered).
    """
    arr = np.sort(lengths)
    half = arr.sum() / 2
    csum = np.cumsum(arr)
    return arr[np.searchsorted(csum, half)]


def add_bin_size_and_N50(bins: Iterable[Bin], contig_to_size: dict[int, int]):
    """
    Add bin size and N50 metrics to a list of bin objects.

    :param bins: List of bin objects.
    :param contig_to_size: Dictionary mapping contig IDs to contig sizes.

    :return: None. The bin objects are updated in place with size and N50.
    """
    # TODO use numpy array everywhere instead of contig_to_size
    contig_sizes = (
        contig_to_size
        if isinstance(contig_to_size, np.ndarray)
        else prepare_contig_sizes(contig_to_size)
    )

    for bin_obj in bins:
        lengths = contig_sizes[
            np.fromiter(bin_obj.contigs, dtype=np.intp)
        ]  # fast bulk lookup
        total_len = lengths.sum()
        n50 = compute_N50(lengths)

        bin_obj.add_length(int(total_len))
        bin_obj.add_N50(int(n50))


def add_bin_coding_density(
    bins: list[Bin], contig_to_coding_length: dict[int, int]
) -> float | None:
    """
    Calculate the coding density of the given bins.

    :param contig_to_coding_length: A dictionary mapping contig IDs to their total coding lengths.

    :return: The coding density of the bin, or None if the length is not set or is zero.
    """
    for bin_obj in bins:
        bin_obj.add_coding_density(contig_to_coding_length)


def add_bin_metrics(
    bins,
    contig_info: dict,
    contamination_weight: float,
    threads: int = 1,
    checkm2_batch_size: int = 500,
    disable_progress_bar: bool = False,
    quality_workers: int | None = None,
    score_cache=None,
):
    """Score bounded membership tasks, updating the parent's candidates in place.

    CPU allocation and model-owner count are separate budgets. Evidence is
    read-only shared mmap; workers return three numeric columns, never Bin objects.
    Optional atomic shards recover completed batches across interruptions.
    """
    import concurrent.futures as cf
    import multiprocessing
    import tempfile
    from pathlib import Path

    from threadpoolctl import threadpool_limits

    from binette.features import ContigEvidence
    from binette.score_store import ScoreStore, scoring_identity
    from binette.scoring import (
        initialize_worker,
        make_processors,
        predict_batch,
        save_reference,
        score_worker,
    )

    if checkm2_batch_size <= 0 or threads <= 0:
        raise ValueError("Batch size and CPU allocation must be positive")
    bins_list = bins if isinstance(bins, list) else list(bins)
    if not bins_list:
        return []
    evidence = contig_info.get("_evidence")
    if evidence is None:
        evidence = ContigEvidence.from_dicts(
            contig_info["contig_to_kegg_counter"],
            contig_info["contig_to_cds_count"],
            contig_info["contig_to_aa_counter"],
            contig_info["contig_to_aa_length"],
        )
    workers = min(threads, 16) if quality_workers is None else quality_workers
    if workers <= 0 or workers > threads:
        raise ValueError("Model workers must be between one and the CPU allocation")
    if quality_workers is None and len(bins_list) <= checkm2_batch_size * 12:
        workers = 1
    store = (
        None
        if score_cache is None
        else ScoreStore(
            Path(score_cache),
            scoring_identity(
                evidence, bins_list, checkm2_batch_size, contamination_weight
            ),
        )
    )

    def apply_result(start, values, persist=True):
        if store is not None and persist:
            store.write(start, values)
        for index, (comp, cont, model) in enumerate(zip(*values, strict=True), start):
            candidate = bins_list[index]
            candidate.add_quality(float(comp), float(cont), contamination_weight)
            candidate._checkm2_model_index = 0 if model else 1

    def tasks():
        for start, batch in membership_batches(bins_list, checkm2_batch_size):
            restored = None if store is None else store.load(start, len(batch))
            if restored is not None:
                apply_result(start, restored, persist=False)
                progress.update(progress_task, advance=len(batch))
            else:
                yield start, [candidate.contigs_key for candidate in batch]

    logger.info(
        "Scoring %s bins with %s persistent model owners", len(bins_list), workers
    )
    with (
        Progress(disable=disable_progress_bar) as progress,
        threadpool_limits(limits=1),
    ):
        progress_task = progress.add_task("Assessing bin quality", total=len(bins_list))
        task_iterator = iter(tasks())
        first = next(task_iterator, None)
        if first is None:
            return bins_list
        if workers == 1:
            prediction, post = make_processors(1)
            for start, keys in chain((first,), task_iterator):
                order = sorted(range(len(keys)), key=keys.__getitem__)
                batch = bins_list[start : start + len(keys)]
                values = predict_batch(
                    evidence,
                    [batch[index].contigs for index in order],
                    prediction,
                    post,
                )
                inverse = np.argsort(order)
                apply_result(start, tuple(value[inverse] for value in values))
                progress.update(progress_task, advance=len(keys))
            return bins_list

        # Spawn avoids inheriting a TensorFlow runtime initialized by original scoring.
        # Persist only numeric evidence in task-local storage; models live per worker.
        with tempfile.TemporaryDirectory(prefix="binette-evidence-") as directory:
            root = Path(directory)
            evidence.save(root)
            save_reference(root)
            with cf.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_worker,
                initargs=(str(root),),
            ) as pool:
                remaining = iter(chain((first,), task_iterator))
                pending = {}
                while True:
                    while len(pending) < 2 * workers:
                        task = next(remaining, None)
                        if task is None:
                            break
                        pending[pool.submit(score_worker, task)] = len(task[1])
                    if not pending:
                        break
                    done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                    for future in done:
                        count = pending.pop(future)
                        start, values = future.result()
                        apply_result(start, values)
                        progress.update(progress_task, advance=count)
    return bins_list


def membership_batches(bins: list, max_bins: int, max_memberships: int = 65536):
    """Bound both model rows and sparse incidence work; keep oversized bins whole."""
    if max_bins <= 0 or max_memberships <= 0:
        raise ValueError("Batch limits must be positive")
    start = 0
    while start < len(bins):
        stop, memberships = start, 0
        while stop < len(bins) and stop - start < max_bins:
            size = len(bins[stop].contigs)
            if stop > start and memberships + size > max_memberships:
                break
            memberships += size
            stop += 1
        yield start, bins[start:stop]
        start = stop


def chunks(iterable, size: int) -> Iterator[tuple]:
    """
    Generate adjacent chunks of data from an iterable.

    :param iterable: The iterable to be divided into chunks.
    :param size: The size of each chunk.
    :return: An iterator that produces tuples of elements in chunks.
    """
    it = iter(iterable)
    return iter(lambda: tuple(islice(it, size)), ())


def balanced_chunks(items: list, num_chunks: int) -> list[list]:
    """
    Distribute items into balanced chunks with more even size distribution.

    :param items: List of items to chunk.
    :param num_chunks: Number of chunks to create.
    :return: List of chunks with balanced sizes.
    """
    if num_chunks <= 0:
        return [items]
    if num_chunks >= len(items):
        return [[item] for item in items]

    # Calculate base size and remainder
    base_size = len(items) // num_chunks
    remainder = len(items) % num_chunks

    chunks_list = []
    start_idx = 0

    for i in range(num_chunks):
        # Some chunks get an extra item to distribute the remainder
        chunk_size = base_size + (1 if i < remainder else 0)
        chunk = items[start_idx : start_idx + chunk_size]
        if chunk:  # Only add non-empty chunks
            chunks_list.append(chunk)
        start_idx += chunk_size

    return chunks_list


def assess_bins_quality(
    bins: Iterable[Bin],
    contig_to_kegg_counter: dict,
    contig_to_cds_count: dict,
    contig_to_aa_counter: dict,
    contig_to_aa_length: dict,
    contamination_weight: float,
    checkm2_batch_size: int,
    postProcessor=None,
    threads: int = 1,
):
    """
    Assess the quality of bins.

    This function assesses the quality of bins based on various criteria and assigns completeness and contamination scores.
    This code is taken from checkm2 and adjusted

    :param bins: List of bin objects.
    :param contig_to_kegg_counter: Dictionary mapping contig names to KEGG counters.
    :param contig_to_cds_count: Dictionary mapping contig names to CDS counts.
    :param contig_to_aa_counter: Dictionary mapping contig names to amino acid counters.
    :param contig_to_aa_length: Dictionary mapping contig names to amino acid lengths.
    :param contamination_weight: Weight for contamination assessment.
    :param postProcessor: A post-processor from checkm2
    :param threads: Number of threads for parallel processing (default is 1).
    :param checkm2_batch_size: Maximum number of bins to process in a single CheckM2 call.
    """
    bins_list = list(bins)

    if not bins_list:
        return []
    if checkm2_batch_size <= 0:
        raise ValueError("checkm2_batch_size must be positive")

    if postProcessor is None:
        modelPostprocessing = get_modelPostprocessing()
        postProcessor = modelPostprocessing.modelProcessor(threads)

    # Prediction models are immutable between batches; load once per scoring call.
    modelProc = get_modelProcessing().modelProcessor(threads)

    # If we have fewer bins than the batch size, process them all at once
    if len(bins_list) <= checkm2_batch_size:
        return _assess_bins_quality_batch(
            bins_list,
            contig_to_kegg_counter,
            contig_to_cds_count,
            contig_to_aa_counter,
            contig_to_aa_length,
            contamination_weight,
            postProcessor,
            threads,
            modelProc=modelProc,
        )

    # Split bins into smaller batches for memory management
    logger.debug(
        f"Splitting {len(bins_list)} bins into batches of {checkm2_batch_size} for CheckM2 processing"
    )

    all_processed_bins = []
    n_batches = (len(bins_list) + checkm2_batch_size - 1) // checkm2_batch_size
    for i, batch_bins in enumerate(chunks(bins_list, checkm2_batch_size)):
        logger.debug(
            f"Processing CheckM2 batch {i + 1}/{n_batches} with {len(batch_bins)} bins"
        )

        # Process this batch
        processed_batch = _assess_bins_quality_batch(
            batch_bins,
            contig_to_kegg_counter,
            contig_to_cds_count,
            contig_to_aa_counter,
            contig_to_aa_length,
            contamination_weight,
            postProcessor,
            threads,
            modelProc=modelProc,
        )

        all_processed_bins.extend(processed_batch)

    return all_processed_bins


def _assess_bins_quality_batch(
    bins: list[Bin],
    contig_to_kegg_counter: dict,
    contig_to_cds_count: dict,
    contig_to_aa_counter: dict,
    contig_to_aa_length: dict,
    contamination_weight: float,
    postProcessor,
    threads: int,
    modelProc=None,
):
    """
    Assess the quality of a batch of bins (internal function).

    This function processes a single batch of bins through CheckM2.
    It's called by assess_bins_quality for each batch when memory management is needed.
    """

    metadata_df = get_bins_metadata_df(
        bins, contig_to_cds_count, contig_to_aa_counter, contig_to_aa_length
    )

    diamond_complete_results, ko_list_length = get_diamond_feature_per_bin_df(
        bins, contig_to_kegg_counter
    )
    diamond_complete_results = diamond_complete_results.drop(columns=["Name"])

    feature_vectors = pd.concat([metadata_df, diamond_complete_results], axis=1)
    feature_vectors = feature_vectors.sort_values(by="Name")

    # Create mapping from bin name to bin object for easy lookup
    bin_name_to_bin = {bin_obj.contigs_key: bin_obj for bin_obj in bins}

    # 4: Call general model & specific models and derive predictions"""
    if modelProc is None:
        modelProc = get_modelProcessing().modelProcessor(threads)

    vector_array = feature_vectors.iloc[:, 1:].values.astype(float)

    logger.debug("Predicting completeness and contamination using the general model")
    general_results_comp, general_results_cont = modelProc.run_prediction_general(
        vector_array
    )

    logger.debug("Predicting completeness using the specific model")
    specific_model_vector_len = (ko_list_length + len(metadata_df.columns)) - 1

    # also retrieve scaled data for CSM calculations
    specific_results_comp, scaled_features = modelProc.run_prediction_specific(
        vector_array, specific_model_vector_len
    )

    logger.debug(
        "Using cosine similarity to reference data to select an appropriate predictor model."
    )

    final_comp, final_cont, models_chosen, csm_array = (
        postProcessor.calculate_general_specific_ratio(
            vector_array[:, 20],
            scaled_features,
            general_results_comp,
            general_results_cont,
            specific_results_comp,
        )
    )

    # Directly iterate through results arrays and lookup corresponding bins
    for bin_name, completeness, contamination, chosen_model in zip(
        feature_vectors["Name"],
        np.round(final_comp, 2),
        np.round(final_cont, 2),
        models_chosen,
        strict=True,
    ):
        bin_obj = bin_name_to_bin[bin_name]
        bin_obj.add_quality(completeness, contamination, contamination_weight)
        bin_obj.add_model(chosen_model)

    return bins
