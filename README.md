This repository accompanies the article "When β Learns to Adapt: Stabilizing Variational Ensembles for Visual Malware Detection" and provides minimal, reproducible examples of the five models described in the study.
Each script in the /src directory corresponds to a distinct model configuration, incrementally extending the baseline CNN architecture with feature-fusion, autoencoding, and adaptive β-VAE mechanisms.

Repository Structure

	src/

   		├── model1.py

   		├── model2.py

   		├── model3.py

   		├── model4.py

   		├── model5.py


	data/sample_malimg/

	README.md




⚙️ Model Descriptions

Model 1 — Baseline CNNs (Single Backbone Evaluation)

This model trains individual pre-trained convolutional networks (VGG16, VGG19, Xception, ResNet50, ConvNeXtLarge) separately on the visual malware datasets.
It establishes a baseline for per-architecture performance and serves as the foundation for the feature-fusion ensemble.
Each CNN is fine-tuned with frozen ImageNet weights, global average pooling, and fully connected classifier layers.

⸻

Model 2 — Feature Fusion Ensemble

Model 2 integrates the feature representations of multiple CNN backbones through concatenation.
Each backbone acts as a parallel feature extractor, and their flattened outputs are merged to form an extended feature space.
This approach enhances feature diversity and improves multi-scale representation learning across malware families.

⸻

Model 3 — Autoencoder-Enhanced Feature Compression

In this configuration, the fused feature vectors obtained from Model 2 are passed through a symmetric autoencoder block.
The encoder compresses the high-dimensional feature space into latent representations (tested with 32, 128, 256 dimensions), while the decoder reconstructs them to enforce information consistency.
This design evaluates how latent-space compression affects generalization and classification stability.

⸻

Model 4 — Fixed-β Variational Autoencoder (β-VAE)

Model 4 replaces the deterministic autoencoder with a variational counterpart, introducing a fixed β coefficient in the KL-divergence term to control the regularization strength.
Three fixed β values (0.05, 0.01, 0.001) are tested across different latent sizes.
While the fixed-β approach promotes disentanglement, it often suffers from posterior collapse and unstable loss dynamics as the feature-space dimensionality increases.

⸻

Model 5 — Annealing-Based β-VAE (Proposed Model)

Model 5 extends Model 4 with an annealing mechanism that dynamically increases β during training, preventing abrupt regularization in early epochs.
This adaptive scheduling stabilizes training, balances reconstruction and regularization terms, and mitigates the exploding-loss problem observed in fixed-β configurations.
Additional refinements include:
	•	Label smoothing (ε = 0.05) to improve calibration across heterogeneous datasets.
	•	Gradient clipping (clipnorm = 1.0) to control gradient explosion.
	•	ReduceLROnPlateau scheduler for adaptive learning rate decay.
	•	Increased dropout (0.3) for stronger regularization in large feature spaces.

⸻
🔗 License

This code is released for academic and research purposes only under the MIT License.
