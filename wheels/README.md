This directory is reserved for binary wheels used by Colab/offline installs.

Expected bundled wheel:

- `sentencepiece-0.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`

The sandbox used for this repair pass could not resolve `files.pythonhosted.org`,
so the wheel could not be downloaded here. Run the following from an environment
with network access before zipping the project:

```bash
python scripts/fetch_wheels.py
```

