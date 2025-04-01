# Microscopy Image Segmentation with Machine Learning  
Work-in-progress project exploring state-of-the-art ML models for nuclei (cell body) segmentation in microscopy images.

![sample-image](link_to_image_or_placeholder.png)


## 🧬 Progress so far: try for yourself!
[demo link]


## 🔍 Status Summary

- Improved mask generation based on *watershed* method (Ref: [topcoders](https://www.kaggle.com/competitions/data-science-bowl-2018/discussion/54741))

- Instance reconstruction using *watershed* method achieves 99% fidelity

- Class imbalance: Minor class (*split-line*) is not learning effectively

Although the current results are still underperforming a baseline reference, the repo documents methods, experiments, learnings, and future directions in this space.

✅ Work in progress  
✅ Research-focused  
✅ Open to feedback and suggestions


## Dive In

### 📁 Dataset

- Name: ![BBBC039](https://bbbc.broadinstitute.org/BBBC039)
- Size: [e.g. 10k+ labeled cell images]
- Type: [e.g. fluorescence microscopy / histopathology slides / phase contrast]
- Format: Images + labels in [CSV/JSON/etc.]

[Include a link to the dataset or data loader code if public.]

---

## 🧪 Current Experiments

| Model              | Augmentation       | Accuracy | F1 Score | Notes                          |
|-------------------|--------------------|----------|----------|--------------------------------|
| ResNet18 (baseline) | Horizontal flips    | 73.2%    | 0.702    | Initial baseline               |
| EfficientNet-B0    | Flip + brightness   | 75.6%    | 0.715    | Slightly better generalization |
| SimCLR (SSL)       | No labels           | TBD      | TBD      | Ongoing training               |

🧠 Ideas in progress:
- Self-supervised pretraining (SimCLR, BYOL)
- Explainability with Grad-CAM
- Cell segmentation with UNet
- Class imbalance handling via focal loss

---

## 📊 Evaluation Metrics

- Accuracy, Precision, Recall, F1 Score
- Confusion matrix
- Grad-CAM visualizations (planned)

---

## 📈 Current Results

📉 So far, none of the trained models outperform a basic [reference model or published baseline], but several hypotheses are being tested:

- Label noise in dataset?
- Underfitting due to resolution?
- Class imbalance?

Model logs, experiments, and plots are tracked in `/notebooks` and `/logs`.

---

## 🛠 How to Run

Basic setup:

```bash
git clone https://github.com/your_username/microscopy-cv-wip.git
cd microscopy-cv-wip
pip install -r requirements.txt
python train.py --config configs/resnet18.yaml
```

To visualize results:

```bash
jupyter notebook notebooks/eval_visualization.ipynb
```
---

## 🧭 Roadmap

- [ ] Improve classification metrics  
- [ ] Add explainability layer (Grad-CAM)  
- [ ] Integrate self-supervised backbone  
- [ ] Segment cells (UNet prototype)  
- [ ] Write blog post summarizing project  
- [ ] Evaluate on external dataset for generalization  
- [ ] Clean up data pipeline for reproducibility  
- [ ] Package model for inference (ONNX or TorchScript)

---

## 🤝 Contributions / Feedback

This project is still exploratory. Feedback, ideas, or PRs are welcome—especially if you're interested in microscopy or biomedical computer vision.

Feel free to:
- Open an issue with suggestions or bugs
- Start a discussion if you’re working on similar problems
- Fork the repo and try your own experiments!

---

## 📚 More Backgrounds & References

- Why interested in nuclei segmentation - logical line
 - Nuclei segmentation remains to be very challenging, especially in histological images (e.g. whole slide image), and useful in various microscopic imaging modalities (fluorecent, bright-field, etc).
 - DSB 2018 was an interesting effort trying to generalize across multiple modalities, but the top-score still leaves big room for improvement, where great advance in recent years could potentially help with (ViT etc.)
 - Start from an "easier" - more homogenus dataset, apply a smart trick borrowed from the top-score solution from DSB 2018. This project aims at to replicate a successful baseline first, then generalized to more complex datasets.



- ["Cell Classification using CNNs" – Smith et al. 2021]  
- [Kaggle: HuBMAP Cell Segmentation Competition](https://www.kaggle.com/competitions/hubmap-organ-segmentation)  
- [Papers with Code: Self-Supervised Learning on Microscopy Images](https://paperswithcode.com/task/microscopy-image-classification)  
- ["Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" – Selvaraju et al. 2017](https://arxiv.org/abs/1610.02391)

---

## 🧾 License

This project is licensed under the MIT License. See the LICENSE file for more details.
