# Input data

`newcomb/newfrat01.csv` through `newfrat15.csv` contain the ranking matrices
used to generate the published Newcomb samples. At each time point, a relationship
is considered present if the rank in either direction is 1–3. The loader also supports
downloading the ZIP archive from the following upstream commit:

```text
https://raw.githubusercontent.com/EngAAlex/MultiDynNos/068aa79680b7d670d2338493bc9c88f4ffbd3db6/data/Newcomb/newfrat.zip
```

```bash
python -m multidynnos_py raw/newcomb -o output/newcomb.json
python -m multidynnos_py --dataset newcomb -o output/newcomb_downloaded.json
```

The existing research inputs—the CSV files in this directory and the `infovis/`
and `vandebunt/` directories—are also retained.
The CLI currently provides a dataset-specific parser for Newcomb.
The original InfoVis and VanDeBunt formats must be converted to a supported event or time-slice JSON format.
When using a three-column event CSV, specify the relationship duration with
`--event-duration`, using the same units as the input timestamps.

See [samples/README.md](../samples/README.md) for commands to reproduce the published samples.
