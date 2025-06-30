# PatchCore

This repository contains a simple implementation of the PatchCore anomaly detection algorithm, which is based on the paper "PatchCore: Efficient Patch-based Outlier Detection" by R. Schneider et al. (2022). The implementation is designed to be straightforward and easy to use, with a focus on clarity and simplicity.
Also the PatchCore classifier is implemented as custom torch module, enabling easy integration into PyTorch-based workflows or export to production environments via ONNX or similar frameworks.

For a reference on how to use the package, please see `example.py`. This script demonstrates typical usage patterns and can serve as a starting point for your own applications.

I plan to eventually publish this package on PyPI for easier usage. Also, as currently the scikit learn and scikit image libraries are only used in evaluating the benchmark asn generating the segmentations i will eventually remove wthem and add custom implementations to compute the AUROC score and find contours.

Benchmark results on the MVTecAD dataset will be uploaded soon.
