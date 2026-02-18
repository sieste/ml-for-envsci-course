keywords = {

    "Machine Learning": 8,
    "Environmental Science": 8,

    # Core ML concepts (highest importance)
    "supervised learning": 6,
    "features": 6,
    "targets": 6,
    "training data": 6,
    "loss function": 6,
    "empirical risk minimisation": 6,
    "generalisation": 6,
    "overfitting": 6,
    "cross validation": 6,
    "benchmarking": 5,
    "skill score": 5,
    "in-sample error": 5,
    "out-of-sample error": 5,

    # Problem types
    "regression": 6,
    "classification": 6,
    "binary classification": 4,
    "multiclass classification": 4,

    # Classical models
    "linear regression": 5,
    "k-nearest neighbours": 4,
    "decision tree": 5,
    "random forest": 5,
    "gradient boosting": 5,
    "xgboost": 4,

    # Tree-specific concepts
    "information gain": 4,
    "gini impurity": 4,
    "entropy": 4,
    "hyperparameters": 4,
    "grid search": 4,

    # Neural networks (very prominent)
    "neural network": 6,
    "multilayer perceptron": 5,
    "activation function": 5,
    "relu": 4,
    "sigmoid": 3,
    "softmax": 4,
    "backpropagation": 5,
    "gradient descent": 5,
    "stochastic gradient descent": 4,
    "adam optimiser": 4,
    "regularisation": 4,
    "dropout": 4,
    "early stopping": 4,

    # Convolutional networks
    "Convolutional Neural Network": 6,
    "Convolution": 5,
    "kernel": 4,
    "padding": 3,
    "pooling": 4,
    "max pooling": 3,
    "flatten": 3,
    "u-net": 4,
    "skip connection": 4,

    # Evaluation metrics
    "mean squared error": 4,
    "accuracy": 4,
    "confusion matrix": 4,
    "cross entropy": 5,

    # Software ecosystem
    "python": 5,
    "scikit-learn": 4,
    "XGBoost": 4,
    "keras": 4,
    "tensorflow": 4,

    # Applications (lower but visible)
    "air quality": 3,
    "remote sensing": 3,
    "radar": 3,
    "ecology": 2,
    "trees": 2,
    "meteorology": 2,
}

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# keyword weights dictionary (from above)
frequencies = keywords

wc = WordCloud(
    width=1600,
    height=800,
    background_color="white",
    colormap="viridis",
    max_words=150,
    relative_scaling=0.5
)

wc.generate_from_frequencies(frequencies)

plt.figure(figsize=(16, 8))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.show()
