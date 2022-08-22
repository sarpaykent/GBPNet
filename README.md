# GBPNet: Universal Geometric Representation Learning on Protein Structures

The structure of the project is as follows:

* `gbpnet.models`: The implementation of the models used in the experiments.
* `gbpnet.datamodules`: The data processing pipeline.
* `run_test.py`: The main script for reproducing the results.

### Installation

The verified versions of the dependencies can be found in `requirements.txt`.

### Data

The datasets are automatically downloaded when the test script is called. The datasets will be stored in `data/`
directory.

### Demo

We provide a demo model in the `models` directory, which can be used to evaluate the results for CPD task.

```bash
python run_test.py ./models/cpd_model_sample.pt cpd
```

### Acknowledgements

The following packages/libraries are adapted/communicated with in the codebase of GBPNet:

- [Pytorch](https://github.com/pytorch/pytorch)
- [Pytorch Lightning](https://www.pytorchlightning.ai/)
- [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [NumPy](https://numpy.org/)
- [Catalyst](https://github.com/catalyst-team/catalyst)
- [Ingraham, et al, NeurIPS 2019](https://github.com/jingraham/neurips19-graph-protein-design)
- [Townshend et al, NeurIPS 2020](https://www.atom3d.ai/)
- [Jing, et al, ICLR 2021](https://github.com/drorlab/gvp-pytorch)

We thank their authors for providing the codebase.


