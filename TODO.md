## A few practical suggestions for phytoplankton classification

- **Image size:** 224×224 is common, but if your dataset is small, 128×128 or 64×64 might reduce overfitting.

- **Augmentations:** rotation, flips, small scaling, brightness changes — but avoid those that break biological shape (e.g., very strong shears).

- **Class imbalance:** phytoplankton datasets often have many rare taxa; consider:
  - Weighted loss functions (e.g., `nn.CrossEntropyLoss(weight=...)`)
  - Oversampling rare classes

- **Metrics:** accuracy is useful, but also track macro F1 or balanced accuracy to see how well you do on rare classes.

- **Model:** small CNN (e.g., ResNet18, EfficientNet-B0) often works very well; ViT can work if you have enough data.

- **Image normalization:** standardize images to have zero mean and unit variance per channel. I think this is using fixed values like `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]` is fine for phytoplankton images. Check if we need to adjust these based on your dataset.
