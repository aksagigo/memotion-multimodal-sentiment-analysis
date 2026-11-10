# Multimodal Meme Sentiment Analysis

## Project Overview

This project investigates sentiment classification of internet memes using both textual and visual information from the Memotion dataset.

Memes often communicate meaning through a combination of images, written text, humour, and sarcasm. This makes sentiment classification challenging when using only one modality.

The project compares four machine learning approaches:

1. Text-only classification using TF-IDF features.
2. Image-only classification using ResNet50 features.
3. Early fusion combining text and image features.
4. Late fusion combining predictions from separate text and image classifiers.

The objective is to determine whether multimodal fusion improves sentiment classification compared with unimodal approaches.

## Dataset

The project uses the Memotion dataset, containing approximately 7,000 memes.

Each meme is associated with an image, textual content, and a sentiment label.

The sentiment classification task uses three categories:

- Positive
- Neutral
- Negative

### Data Splitting

| Dataset | Samples |
|---|---:|
| Training | 4,894 |
| Validation | 1,048 |
| Testing | 1,050 |
| Total | 6,992 |

### Data Preprocessing and Quality

- Prepared meme text for TF-IDF feature extraction.
- Extracted visual representations from meme images.
- Checked image readability and handled problematic image files.
- Maintained separate training, validation, and test datasets.
- Examined class distribution and classification errors.

The dataset is imbalanced, with positive memes more frequent than negative memes. This is an important consideration when interpreting model performance.

## Methodology

### 1. Text-Only Model

TF-IDF (Term Frequency–Inverse Document Frequency) was used to transform meme text into numerical features.

A machine learning classifier was trained to predict sentiment using textual information alone.

### 2. Image-Only Model

A pretrained ResNet50 convolutional neural network was used to extract visual features from meme images.

Each image was represented using a 2,048-dimensional feature vector, which was then used for sentiment classification.

### 3. Early Fusion

Early fusion combines text and image representations before classification.

TF-IDF text features and ResNet50 image features were concatenated into a single representation.

The combined feature vectors contained 7,048 dimensions.

A classifier was trained using these multimodal features.

### 4. Late Fusion

Late fusion combines the prediction probabilities of the independently trained text-only and image-only classifiers.

A weighted combination was used to generate the final sentiment predictions.

Fusion weights and model hyperparameters were selected using validation data rather than the held-out test set.

## Model Evaluation

Models were evaluated using multiple metrics:

- Accuracy
- Macro F1-score
- Weighted F1-score
- Balanced accuracy
- Macro ROC-AUC
- Confusion matrices
- Classification error analysis

Macro F1 was particularly important because the dataset contains unequal numbers of examples across sentiment classes.

## Test Results

| Model | Accuracy | Macro F1 | Weighted F1 | Balanced Accuracy | Macro ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Text-only (TF-IDF) | 0.4152 | 0.3261 | 0.4321 | 0.3283 | 0.4937 |
| **Image-only (ResNet50)** | **0.4238** | **0.3351** | **0.4361** | **0.3383** | **0.5155** |
| Early Fusion | 0.4219 | 0.3327 | 0.4343 | 0.3356 | 0.5152 |
| Late Fusion | 0.4219 | 0.3318 | 0.4340 | 0.3344 | 0.4975 |

### Key Findings

The image-only model achieved the highest test accuracy (42.38%) and macro F1-score (0.3351).

Early fusion and late fusion performed similarly but did not outperform the image-only baseline.

The differences between the approaches were relatively small.

These findings suggest that combining text and image features does not automatically improve meme sentiment classification. The effectiveness of fusion depends on the quality of the individual representations and the relationships between modalities.

## Overfitting Assessment

Training and validation performance were compared to investigate overfitting.

| Model | Training Accuracy | Validation Accuracy | Training Macro F1 | Validation Macro F1 |
|---|---:|---:|---:|---:|
| Text-only | 0.7869 | 0.4323 | 0.7680 | 0.3533 |
| Image-only | 0.9011 | 0.4351 | 0.9223 | 0.3487 |
| Early Fusion | 0.9461 | 0.4389 | 0.9571 | 0.3456 |

All three evaluated models demonstrated substantial differences between training and validation performance.

Early fusion exhibited the largest training–validation gap, suggesting considerable overfitting.

This indicates that stronger regularisation, improved feature representations, or additional training data may help improve generalisation.

## Visualisations

### Confusion Matrices

![Confusion matrices](confusion_matrices.png)

### Error Analysis

![Error analysis](error_analysis.png)

### Misclassified Meme Examples

![Misclassified memes](misclassified_memes%20%281%29.png)

These visualisations help identify common classification errors and illustrate the difficulty of interpreting sentiment from memes.

## Limitations

- The sentiment classes are imbalanced.
- Humour, sarcasm, and cultural references are difficult to interpret.
- TF-IDF does not fully capture contextual or semantic meaning.
- ResNet50 visual features may not capture the relationship between image content and embedded text.
- The models demonstrate substantial overfitting.
- Multimodal fusion did not produce a meaningful improvement over the strongest unimodal baseline.

## Ethics and Bias Considerations

Memes may contain stereotypes, offensive language, culturally specific humour, and potentially harmful representations.

The dataset may reflect biases in online content and its sentiment annotations.

Class imbalance may also result in poorer performance for underrepresented sentiment categories.

Model predictions should therefore be interpreted cautiously, particularly when analysing culturally sensitive or ambiguous content.

## Future Improvements

Potential improvements include:

- Using contextual text embeddings from transformer-based language models.
- Exploring stronger visual encoders.
- Investigating attention-based multimodal fusion.
- Applying regularisation to reduce overfitting.
- Addressing class imbalance through appropriate training strategies.
- Evaluating model performance across different meme categories.

## Repository Files

| File | Description |
|---|---|
| `Copy_of_Memotion_Multimodal_Project (2).ipynb` | Google Colab experiment notebook |
| `test_results.csv` | Final model evaluation metrics |
| `confusion_matrices.png` | Model confusion matrices |
| `error_analysis.png` | Classification error analysis |
| `misclassified_memes (1).png` | Examples of incorrectly classified memes |
| `README.md` | Project documentation |

*Note: The notebook filename should be updated here if the final version uses a different name.*

## Reproducibility

The project was developed using Python and Google Colab.

Main libraries include:

- pandas and NumPy
- scikit-learn
- PyTorch
- torchvision
- Matplotlib

To reproduce the experiments:

1. Download or clone this repository.
2. Obtain the Memotion dataset from its authorised source.
3. Open the project notebook in Google Colab.
4. Install any required Python dependencies.
5. Configure dataset paths according to the notebook.
6. Execute the notebook cells in sequence to preprocess the data, extract features, train the models, and evaluate performance.

The repository contains the experiment notebook and evaluation outputs. Reproduction requires access to the original dataset and the appropriate computational environment.

## Conclusion

This project compared text-only, image-only, early fusion, and late fusion approaches for three-class meme sentiment classification.

The image-only ResNet50-based model achieved the strongest overall test performance, while both multimodal fusion approaches produced comparable but slightly lower results.

The experiments demonstrate the challenges of multimodal sentiment analysis, including class imbalance, ambiguous sentiment, and overfitting.

Future work could investigate more advanced language and vision representations to improve classification performance.
