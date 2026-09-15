"""Write a stream of Arrow tables into numbered parquet files of ~N jets each.

Used by two entry points, which share this so that a rechunked dataset and a
freshly processed one come out identical:

  - scripts/rechunk.py    : rechunk .parquet files that already exist
  - scripts/ntupelize_list.py : fill the chunks straight from the ROOT input

An ALEPH ntuple is either jet-level (one row per jet) or event-level (one row
per event, with the jets of the event in a list column).  A row is never split
across output files, so an event-level file holds *about* `jets_per_file` jets,
landing on a whole-event boundary, while a jet-level file holds exactly that
many rows.
"""

import os

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
    ):
        self.output_dir = output_dir
        self.prefix = prefix
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
        path = self.path(len(self.paths))
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
