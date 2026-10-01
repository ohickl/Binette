import concurrent.futures as cf
import gzip
import io
import logging
from collections import Counter, defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyfastx
import pyrodigal

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProteinSummary:
    cds_count: int
    amino: Counter
    aa_length: int


def get_contig_from_cds_name(cds_name: str) -> str:
    """
    Extract the contig name from a CDS name.

    :param cds_name: The name of the CDS.
    :type cds_name: str
    :return: The name of the contig.
    :rtype: str
    """
    return "_".join(cds_name.split("_")[:-1])


def predict(
    contigs_iterator: Iterator, outfaa: str, threads: int = 1, summarize: bool = False
) -> tuple[dict[str, list[str]], dict[str, int | None]]:
    """
    Predict open reading frames with Pyrodigal.

    :param contigs_iterator: An iterator of contig sequences.
    :param outfaa: The output file path for predicted protein sequences (in FASTA format).
    :param threads: Number of CPU threads to use (default is 1).

    :return: A dictionary mapping contig names to predicted genes and a dictionary mapping contig names to coding lengths.
    """

    orf_finder = pyrodigal.GeneFinder(meta="meta")

    logger.info(f"Predicting CDS sequences with Pyrodigal using {threads} threads")

    if threads <= 0:
        raise ValueError("Gene-prediction workers must be positive")
    contig_to_genes, contig_to_coding_length = {}, {}
    remaining = iter(contigs_iterator)
    with (
        cf.ThreadPoolExecutor(max_workers=threads) as pool,
        gzip.open(outfaa, "wt") as output,
    ):
        pending = deque()
        while True:
            while len(pending) < 2 * threads:
                item = next(remaining, None)
                if item is None:
                    break
                pending.append(pool.submit(predict_genes, orf_finder.find_genes, *item))
            if not pending:
                break
            contig_id, genes = pending.popleft().result()
            # Write upstream's exact headers/wrapping and translate only once.
            buffer = io.StringIO()
            genes.write_translations(buffer, contig_id)
            text = buffer.getvalue()
            output.write(text)
            sequences = []
            pieces = []
            for line in text.splitlines():
                if line.startswith(">"):
                    if pieces:
                        sequences.append("".join(pieces))
                    pieces = []
                else:
                    pieces.append(line.strip())
            if pieces:
                sequences.append("".join(pieces))
            if summarize:
                amino = get_aa_composition(sequences)
                contig_to_genes[contig_id] = ProteinSummary(
                    len(genes), amino, sum(amino.values())
                )
            else:
                contig_to_genes[contig_id] = sequences
            contig_to_coding_length[contig_id] = get_contig_coding_len(
                genes, len(genes.sequence)
            )

    return contig_to_genes, contig_to_coding_length


def get_contig_coding_len(
    genes: list[pyrodigal.Gene], contig_length: int
) -> int | None:
    """
    Compute the coding length of a contig. Use a mask to account for overlapping genes.

    :param genes: A list of gene annotations for the contig.
    :param contig_length: The length of the contig in base pairs.
    :return: The coding length as a float, or None if contig_length is zero.
    """
    if contig_length == 0:
        return None
    # Merge half-open intervals: no per-base float64/bool allocation is needed.
    intervals = sorted(
        slice(g.begin - 1, g.end).indices(contig_length)[:2] for g in genes
    )
    covered = 0
    right = 0
    for begin, end in intervals:
        if end > max(begin, right):
            covered += end - max(begin, right)
            right = end
    return covered


def predict_genes(find_genes, name, seq) -> tuple[str, pyrodigal.Genes]:
    return (name, find_genes(seq))


def write_faa(outfaa: str, contig_to_genes: list[tuple[str, pyrodigal.Genes]]) -> None:
    """
    Write predicted protein sequences to a FASTA file.

    :param outfaa: The output file path for predicted protein sequences (in FASTA format).
    :param contig_to_genes: A dictionary mapping contig names to predicted genes.

    """
    logger.info("Writing predicted protein sequences")
    with gzip.open(outfaa, "wt") as fl:
        for contig_id, genes in contig_to_genes:
            genes.write_translations(fl, contig_id)


def is_nucleic_acid(sequence: str) -> bool:
    """
    Determines whether the given sequence is a DNA or RNA sequence.

    :param sequence: The sequence to check.
    :return: True if the sequence is a DNA or RNA sequence, False otherwise.
    """
    # Define nucleotidic bases (DNA and RNA)
    nucleotidic_bases = set("ATCGNUatcgnu")

    # Check if all characters in the sequence are valid nucleotidic bases (DNA or RNA)
    if all(base in nucleotidic_bases for base in sequence):
        return True

    # If any character is invalid, return False
    return False


def parse_faa_file(faa_file: str, summarize: bool = False) -> dict:
    """
    Parse a FASTA file containing protein sequences and organize them by contig.

    :param faa_file: Path to the input FASTA file.
    :return: A dictionary mapping contig names to lists of protein sequences.
    :raises ValueError: If the file contains nucleotidic sequences instead of protein sequences.
    """
    contig_to_genes = defaultdict(list)
    checked_sequences = []

    # Iterate through the FASTA file and parse sequences
    for name, seq in pyfastx.Fastx(faa_file):
        contig = get_contig_from_cds_name(name)
        if summarize:
            previous = contig_to_genes.get(contig)
            amino = Counter() if previous is None else previous.amino
            amino.update(seq)
            amino.pop("*", None)
            contig_to_genes[contig] = ProteinSummary(
                1 if previous is None else previous.cds_count + 1,
                amino,
                sum(amino.values()),
            )
        else:
            contig_to_genes[contig].append(seq)

        # Concatenate up to the first 20 sequences for validation
        if len(checked_sequences) < 20:
            checked_sequences.append(seq)

    # Concatenate all checked sequences for a more reliable nucleic acid check
    concatenated_seq = "".join(checked_sequences)

    # Check if the concatenated sequence appears to be nucleic acid
    if is_nucleic_acid(concatenated_seq):
        raise ValueError(
            f"The file '{faa_file}' appears to contain nucleotide sequences. "
            "Ensure that the file contains valid protein sequences in FASTA format."
        )

    return dict(contig_to_genes)


def get_aa_composition(genes: list[str]) -> Counter:
    """
    Calculate the amino acid composition of a list of protein sequences.

    :param genes: A list of protein sequences.
    :return: A Counter object representing the amino acid composition.
    """
    aa_counter = Counter()
    for gene in genes:
        aa_counter.update(gene)
    # remove * from Counter
    aa_counter.pop("*", None)
    return aa_counter


def get_contig_cds_metadata_flat(
    contig_to_genes: dict[str, list[str]],
) -> tuple[dict[str, int], dict[str, Counter], dict[str, int]]:
    """
    Calculate metadata for contigs, including CDS count, amino acid composition, and total amino acid length.

    :param contig_to_genes: A dictionary mapping contig names to lists of protein sequences.
    :return: A tuple containing dictionaries for CDS count, amino acid composition, and total amino acid length.
    """
    contig_to_cds_count = {
        contig: len(genes) for contig, genes in contig_to_genes.items()
    }

    contig_to_aa_counter = {
        contig: get_aa_composition(genes) for contig, genes in contig_to_genes.items()
    }
    logger.info("Calculating amino acid composition")

    contig_to_aa_length = {
        contig: sum(counter.values())
        for contig, counter in contig_to_aa_counter.items()
    }
    logger.info("Calculating total amino acid length")

    return contig_to_cds_count, contig_to_aa_counter, contig_to_aa_length


def get_contig_cds_metadata(
    contig_to_genes: dict[int, Any | list[Any]], threads: int
) -> dict[str, dict]:
    """
    Calculate metadata for contigs in parallel, including CDS count, amino acid composition, and total amino acid length.

    :param contig_to_genes: A dictionary mapping contig names to lists of protein sequences.
    :param threads: Number of CPU threads to use.
    :return: A tuple containing dictionaries for CDS count, amino acid composition, and total amino acid length.
    """
    if contig_to_genes and all(
        isinstance(item, ProteinSummary) for item in contig_to_genes.values()
    ):
        return {
            "contig_to_cds_count": {
                key: item.cds_count for key, item in contig_to_genes.items()
            },
            "contig_to_aa_counter": {
                key: item.amino for key, item in contig_to_genes.items()
            },
            "contig_to_aa_length": {
                key: item.aa_length for key, item in contig_to_genes.items()
            },
        }
    contig_to_cds_count = {
        contig: len(genes) for contig, genes in contig_to_genes.items()
    }

    completed_compositions = {}
    remaining_contigs = iter(contig_to_genes.items())
    logger.info(f"Collecting contig amino acid composition using {threads} threads")
    with cf.ProcessPoolExecutor(max_workers=threads) as tpe:
        pending = {}
        # Keep workers fed without retaining one Future and queued payload per contig.
        while True:
            while len(pending) < 2 * threads:
                try:
                    contig, genes = next(remaining_contigs)
                except StopIteration:
                    break
                pending[tpe.submit(get_aa_composition, genes)] = contig
            if not pending:
                break
            done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for future in done:
                contig = pending.pop(future)
                completed_compositions[contig] = future.result()

    contig_to_aa_counter = {
        contig: completed_compositions[contig] for contig in contig_to_genes
    }
    logger.info("Calculating amino acid composition in parallel")

    contig_to_aa_length = {
        contig: sum(counter.values())
        for contig, counter in contig_to_aa_counter.items()
    }
    logger.info("Calculating total amino acid length in parallel")

    contig_info = {
        "contig_to_cds_count": contig_to_cds_count,
        "contig_to_aa_counter": contig_to_aa_counter,
        "contig_to_aa_length": contig_to_aa_length,
    }

    return contig_info


def filter_faa_file(
    contigs_to_keep: set[str],
    input_faa_file: Path,
    filtered_faa_file: Path,
):
    """
    Filters a FASTA file containing protein sequences to only include sequences
    from contigs present in the provided set of contigs (`contigs_to_keep`).

    This function processes the input FASTA file, identifies protein sequences
    originating from contigs listed in `contigs_to_keep`, and writes the filtered
    sequences to a new FASTA file. The output file supports optional `.gz` compression.

    :param contigs_to_keep: A set of contig names to retain in the output FASTA file.
    :param input_faa_file: Path to the input FASTA file containing protein sequences.
    :param filtered_faa_file: Path to the output FASTA file for filtered sequences.
                              If the filename ends with `.gz`, the output will be compressed.
    """
    # Determine whether the output file should be compressed
    proper_open = gzip.open if str(filtered_faa_file).endswith(".gz") else open

    # Initialize tracking sets for metrics
    contigs_with_genes = set()
    contigs_parsed = set()

    # Process the input FASTA file and filter sequences based on contigs_to_keep
    with proper_open(filtered_faa_file, "wt") as fl:
        for name, seq in pyfastx.Fastx(input_faa_file):
            contig = get_contig_from_cds_name(name)
            contigs_parsed.add(contig)
            if contig in contigs_to_keep:
                contigs_with_genes.add(contig)
                fl.write(f">{name}\n{seq}\n")

    # Calculate metrics
    total_contigs = len(contigs_to_keep)
    contigs_with_no_genes = total_contigs - len(contigs_with_genes)
    contigs_not_in_keep_list = len(contigs_parsed - set(contigs_to_keep))

    # Log the computed metrics
    logger.info(f"Processing protein sequences from '{input_faa_file}'")
    logger.info(
        f"Filtered '{input_faa_file}' to retain genes from {total_contigs} contigs that are included in the input bins"
    )
    logger.debug(
        f"Found {contigs_with_no_genes}/{total_contigs} contigs ({contigs_with_no_genes / total_contigs:.2%}) with no genes"
    )
    logger.debug(
        f"{contigs_not_in_keep_list} contigs from the input FASTA file are not in the keep list"
    )
