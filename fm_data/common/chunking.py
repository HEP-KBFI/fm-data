"""Write a stream of Arrow tables into numbered parquet files of ~N jets each.

Used by `rechunk` (scripts/rechunk.py, and mode=rechunk of scripts/ntupelize.py, which rechunks
the per-task jet tables of each sample).

A table is either jet-level (one row per jet) or event-level (one row per event,
with the jets of the event in a list column).  A row is never split
across output files, so an event-level file holds *about* `jets_per_file` jets,
landing on a whole-event boundary, while a jet-level file holds exactly that
many rows.
"""

import glob
import os
import re

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Number of jets per output file, and the width of the zero-padded file index.
# Padding keeps a plain lexicographic listing in chunk order.
DEFAULT_JETS_PER_FILE = 100_000
CHUNK_INDEX_WIDTH = 5

# Rows per parquet row group; matches what the ALEPH ntupelizer has always used.
DEFAULT_ROW_GROUP_SIZE = 1024

# The field whose length says how many jets a row holds.
JET_FIELD = "jet_pt"


def plain_schema(schema):
    """Strip field and schema metadata so schemas from different files compare.

    Parquet written by awkward carries per-field metadata (ak:parameters and
    friends) that can differ between files whose field *types* are identical,
    which would make ParquetWriter reject the second file.
    """
    return pa.schema([field.with_metadata(None) for field in schema], metadata=None)


def is_event_level(schema, jet_field=JET_FIELD):
    """True if a row is an event (jets in a list column), False if it is a jet."""
    if jet_field not in schema.names:
        raise KeyError(
            f"No {jet_field!r} field in the input; cannot tell how many jets a row "
            f"holds. Available fields: {schema.names}"
        )
    field_type = schema.field(jet_field).type
    return pa.types.is_list(field_type) or pa.types.is_large_list(field_type)


def jets_per_row(table, jet_field=JET_FIELD):
    """Number of jets in each row of a table."""
    column = table.column(jet_field)
    if is_event_level(table.schema, jet_field):
        return np.asarray(pc.list_value_length(column).to_numpy(zero_copy_only=False))
    return np.ones(table.num_rows, dtype=np.int64)


class JetChunkWriter:
    """Fill numbered parquet files with `jets_per_file` jets each.

    Tables are added in stream order and buffered until a file's worth has
    gathered, then written in a single call so that the row groups all come out
    the configured size.  Memory is therefore bounded by one output file, not by
    the size of the dataset.
    """

    def __init__(
        self,
        output_dir,
        prefix,
        schema,
        jets_per_file=DEFAULT_JETS_PER_FILE,
        row_group_size=DEFAULT_ROW_GROUP_SIZE,
        jet_field=JET_FIELD,
        first_index=0,
    ):
        self.output_dir = output_dir
        self.prefix = prefix
        # index of the first file written, to continue a numbered series
        self.first_index = int(first_index)
        self.schema = plain_schema(schema)
        self.jets_per_file = int(jets_per_file)
        self.row_group_size = int(row_group_size)
        self.jet_field = jet_field
        if self.jets_per_file < 1:
            raise ValueError(f"jets_per_file must be >= 1, got {jets_per_file}")
        self.paths = []
        self.total_rows = 0
        self.total_jets = 0
        self._buffer = []
        self._rows_buffered = 0
        self._jets_buffered = 0
        os.makedirs(output_dir, exist_ok=True)

    def path(self, index):
        return os.path.join(
            self.output_dir, f"{self.prefix}_{index:0{CHUNK_INDEX_WIDTH}d}.parquet"
        )

    def add(self, table):
        """Add a table, closing off an output file whenever one is full."""
        table = table.cast(self.schema)
        counts = jets_per_row(table, self.jet_field)
        offset = 0
        while offset < table.num_rows:
            room = self.jets_per_file - self._jets_buffered
            cumulative = np.cumsum(counts[offset:])
            # Rows that still fit without going over the target.  Rows are never
            # split, so the boundary lands between two rows.
            n_rows = int(np.searchsorted(cumulative, room, side="right"))
            if n_rows == 0:
                if self._rows_buffered > 0:
                    # No room left in the file being filled: close it and retry
                    # the same row against an empty one.
                    self._write()
                    continue
                # A single row holding more jets than a whole file: keep it in
                # one piece rather than loop forever.
                n_rows = 1
            self._buffer.append(table.slice(offset, n_rows))
            self._rows_buffered += n_rows
            self._jets_buffered += int(cumulative[n_rows - 1])
            offset += n_rows
            if self._jets_buffered >= self.jets_per_file:
                self._write()

    def _write(self):
        """Write everything buffered as one output file."""
        if not self._buffer:
            return
        path = self.path(self.first_index + len(self.paths))
        table = pa.concat_tables(self._buffer)
        with pq.ParquetWriter(
            path,
            self.schema,
            compression="zstd",
            compression_level=6,
            use_byte_stream_split=True,
        ) as writer:
            writer.write_table(table, row_group_size=self.row_group_size)
        self.paths.append(path)
        self.total_rows += self._rows_buffered
        self.total_jets += self._jets_buffered
        print(
            f"  -> {path} ({self._rows_buffered} rows, {self._jets_buffered} jets)",
            flush=True,
        )
        self._buffer = []
        self._rows_buffered = 0
        self._jets_buffered = 0

    def close(self):
        """Flush the partially filled last file and return every path written."""
        self._write()
        return self.paths


def natural_key(path):
    """Sort key ordering any numbers in a filename numerically."""
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", os.path.basename(path))
    )


def resolve_inputs(input_loc):
    """Return the input .parquet files: a list of files (kept in its order), a directory, a
    glob, or a single file."""
    if isinstance(input_loc, (list, tuple)):
        missing = [path for path in input_loc if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} input files missing, e.g. {missing[0]}")
        return list(input_loc)
    if os.path.isdir(input_loc):
        paths = glob.glob(os.path.join(input_loc, "*.parquet"))
    elif "*" in input_loc:
        paths = glob.glob(input_loc)
    elif os.path.isfile(input_loc):
        paths = [input_loc]
    else:
        raise FileNotFoundError(f"No such file, directory or glob: {input_loc}")
    if not paths:
        raise FileNotFoundError(f"No .parquet files found in {input_loc}")
    return sorted(paths, key=natural_key)


def build_row_filter(max_abs_eta=None, min_jet_energy=None):
    """Return a callable filtering a jet-level table by the given cuts, or None."""
    if max_abs_eta is None and min_jet_energy is None:
        return None

    def row_filter(table):
        mask = None
        if max_abs_eta is not None:
            mask = pc.less(pc.abs(table.column("jet_eta")), max_abs_eta)
        if min_jet_energy is not None:
            above = pc.greater(table.column("jet_energy"), min_jet_energy)
            mask = above if mask is None else pc.and_(mask, above)
        return table.filter(mask)

    return row_filter


def rechunk(
    input_loc,
    output_dir,
    prefix="data",
    jets_per_file=DEFAULT_JETS_PER_FILE,
    row_group_size=DEFAULT_ROW_GROUP_SIZE,
    jet_field=JET_FIELD,
    max_abs_eta=None,
    min_jet_energy=None,
):
    """Stream the inputs into numbered files of jets_per_file jets ea

    max_abs_eta / min_jet_energy, if given, additionally drop jets failing
    |jet_eta| < max_abs_eta and/or jet_energy > min_jet_energy while rechunking
    (jet-level input only -- see module docstring).
    """
    input_paths = resolve_inputs(input_loc)
    schema = plain_schema(pq.ParquetFile(input_paths[0]).schema_arrow)
    level = "event" if is_event_level(schema, jet_field) else "jet"
    row_filter = build_row_filter(max_abs_eta, min_jet_energy)
    if row_filter is not None and level == "event":
        raise ValueError(
            "max_abs_eta / min_jet_energy filtering only supports jet-level "
            f"input (one row per jet); {input_paths[0]} is event-level."
        )
    filter_desc = " and ".join(
        s
        for s in [
            f"|jet_eta| < {max_abs_eta}" if max_abs_eta is not None else None,
            f"jet_energy > {min_jet_energy}" if min_jet_energy is not None else None,
        ]
        if s is not None
    )
    print(
        f"Rechunking {len(input_paths)} {level}-level file(s) into "
        f"{jets_per_file}-jet files under {output_dir}"
        + (f" (filtering: {filter_desc})" if filter_desc else "")
    )

    writer = JetChunkWriter(
        output_dir=output_dir,
        prefix=prefix,
        schema=schema,
        jets_per_file=jets_per_file,
        row_group_size=row_group_size,
        jet_field=jet_field,
    )
    # Writing into the input directory is fine, writing over the inputs is not:
    # the input list is fixed up front, so a clash would silently corrupt a file
    # that has not been read yet.
    output_pattern = re.compile(rf"^{re.escape(prefix)}_\d+\.parquet$")
    clashing = [
        path
        for path in input_paths
        if os.path.dirname(os.path.abspath(path)) == os.path.abspath(output_dir)
        and output_pattern.match(os.path.basename(path))
    ]
    if clashing:
        raise ValueError(
            f"The output would overwrite its own input, e.g. {clashing[0]}. "
            f"Use a different --output-dir or --prefix."
        )

    n_rows_in = 0
    n_rows_kept = 0
    for i, path in enumerate(input_paths, 1):
        parquet_file = pq.ParquetFile(path)
        if parquet_file.metadata.num_rows == 0:
            print(f"[{i}/{len(input_paths)}] {os.path.basename(path)}: empty, skipping")
            continue
        print(f"[{i}/{len(input_paths)}] {os.path.basename(path)}", flush=True)
        for row_group in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(row_group).replace_schema_metadata(None)
            n_rows_in += table.num_rows
            if row_filter is not None:
                table = row_filter(table)
            n_rows_kept += table.num_rows
            if table.num_rows > 0:
                writer.add(table)
    paths = writer.close()

    if writer.total_rows != n_rows_kept:
        raise RuntimeError(
            f"Wrote {writer.total_rows} rows but kept {n_rows_kept} after reading "
            f"{n_rows_in}; refusing to report success on a dataset that lost or "
            f"duplicated rows."
        )
    dropped_msg = f" ({n_rows_in - n_rows_kept} dropped by the filter)" if row_filter is not None else ""
    print(
        f"Done: {n_rows_in} rows read{dropped_msg}, {n_rows_kept} kept / "
        f"{writer.total_jets} jets from {len(input_paths)} file(s) into "
        f"{len(paths)} file(s)"
    )
    return paths
