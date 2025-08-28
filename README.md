# **MD4‑MS/MS: Masked Diffusion for Tandem‑MS Structural Elucidation**

*A fork of **MD4: Simplified and Generalized Masked Diffusion for Discrete Data**  
extended to large‑scale pre‑training on SMILES, conditioned on **Morgan
fingerprints predicted from MS/MS spectra with the MIST model**.*

---

## Prerequisite

Create a fresh environment (with libtpu):

```bash
uv sync
```

## Overview

  - Large scale, Multihost training capable
  - DDP, FSDP, TP implemented using jax manual parallelism
  - Zero 0 for the moment, Zero 1 and 2 planned (should be easy to do) 


---

## Acknowledgement 
```
@inproceedings{shi2024simplified,
  title={Simplified and Generalized Masked Diffusion for Discrete Data},
  author={Shi, Jiaxin and Han, Kehang and Wang, Zhe and Doucet, Arnaud and Titsias, Michalis K.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}

Goldman, S., Wohlwend, J., Stražar, M. et al. Annotating metabolite mass spectra with domain-inspired chemical formula transformers. Nat Mach Intell (2023). https://doi.org/10.1038/s42256-023-00708-3
```