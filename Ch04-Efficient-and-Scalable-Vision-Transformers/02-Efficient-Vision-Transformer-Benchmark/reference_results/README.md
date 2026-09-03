# Reference results policy

This directory intentionally contains no prefilled accuracy or throughput values. Publishing
plausible-looking numbers that were not produced by the checked-in protocol would undermine the
chapter's central lesson about fair benchmarking.

To add verified reference results:

1. run `configs/full.yaml` to completion;
2. run the inference benchmark once on the declared reference device;
3. generate the report;
4. copy the generated report and CSV files here without hand-editing numeric cells;
5. include `benchmark.json`, the preset fingerprint, split checksum, device metadata, command, and
   execution date; and
6. have a second person or automated job verify that every configured run is complete.

Quick and smoke artifacts must never be published as reference results.
