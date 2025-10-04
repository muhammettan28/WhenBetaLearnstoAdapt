This repository accompanies the article When β Learns to Adapt: Stabilizing Variational Ensembles for Visual Malware Detection
 and provides minimal, reproducible examples of the five models described in the study.
Each script in the /src directory corresponds to a distinct model configuration, incrementally extending the baseline CNN architecture with feature-fusion, autoencoding, and adaptive β-VAE mechanisms.

Repository Structure
├── src/
│   ├── model1.py
│   ├── model2.py
│   ├── model3.py
│   ├── model4.py
│   ├── model5.py
│
├── data/
│   └── sample_malimg/        # small subset (e.g., 10 benign, 10 malware images)
│
└── README.md
