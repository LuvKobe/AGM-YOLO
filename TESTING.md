# Testing

The repository includes unit tests for the AIC, GCE, and MSA modules.

## Run the test suite

Install the development dependencies and execute:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The tests cover:

- input/output tensor shapes;
- AIC calibration behavior;
- GCE channel, width, and height attention branches;
- MSA feature alignment and selective aggregation;
- gradient propagation through inputs and learnable parameters;
- state-dictionary save/load consistency;
- invalid input handling;
- non-square and non-integer scale ratios; and
- optional CUDA mixed-precision execution when a CUDA device is available.

## Example execution

```bash
python example.py --device cpu
```

For a CUDA-enabled PyTorch environment:

```bash
python example.py --device cuda
```

The GitHub Actions workflow runs the CPU test suite on pushes and pull requests.
