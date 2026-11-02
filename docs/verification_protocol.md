# Series provenance and verification protocol

`series_source_audit.csv` provides row-level candidate identifiers, dates, opponents, scores, and external source URLs. Its `external_status` column is **legacy annotation**, not substantiation that independent checking occurred. URLs alone do not prove a specific row was checked.

For a future manual verification, independently open each listed source and record: (1) event and stage, (2) official date, (3) two competitors, (4) scheduled format Bo3/Bo5 independent of realised score, (5) final series score, (6) source URL, (7) reviewer and review timestamp, (8) whether all constituent games are present. Retain discrepancies and unresolved rows instead of assigning success to an entire competition at once.

`series_exploratory_backtest.csv` is currently **exploratory only**. It uses first-game pre-match Elo with zero pre-series side bonus. It must not be described as independently verified by published match results. The 33 selected series should not be treated as a representative evaluation population.
