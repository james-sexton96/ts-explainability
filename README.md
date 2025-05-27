# Time Series Prediction and Explainability with Transformers

This repository implements a Transformer-based model for time series prediction with built-in explainability features. The model is specifically designed for financial time series data, with a focus on stock price prediction and interpretation of the model's decisions.

## Project Overview

The project consists of two main components:
1. A custom Transformer model for time series prediction (`simple_transformer_ts_classification.py`)
2. An explainability module for interpreting model predictions (`ts_explain.py`)

### Features

- **Advanced Time Series Features**:
  - Technical indicators (RSI, MACD, EMA)
  - Price-based features (returns, momentum)
  - Volatility measures
  - Moving averages

- **Model Architecture**:
  - Custom Transformer implementation with:
    - Multi-head self-attention
    - Positional encoding
    - Batch normalization
    - Dropout for regularization
    - Classification head for binary prediction

- **Training Pipeline**:
  - Automatic feature engineering
  - Data scaling and normalization
  - Feature selection using L1 regularization
  - Hyperparameter tuning
  - Early stopping
  - Model checkpointing

- **Explainability**:
  - Feature importance visualization
  - Time-series saliency maps
  - Model interpretation tools

## Requirements

The project requires the following main dependencies:
- PyTorch
- NumPy
- Pandas
- scikit-learn
- yfinance
- TSInterpret
- Matplotlib

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure

- `simple_transformer_ts_classification.py`: Main model implementation and training pipeline
- `ts_explain.py`: Model explainability and interpretation tools
- `data/`: Directory containing saved model data and scalers
  - `train_data.npz`: Training dataset
  - `test_data.npz`: Test dataset
  - `scalers.pkl`: Fitted feature scalers
- `final_model.pth`: Trained model weights
- `explanation_plot.png`: Visualization of feature importance

## Usage

1. **Training the Model**:
   ```python
   python simple_transformer_ts_classification.py
   ```
   This will:
   - Download and prepare financial data
   - Engineer features
   - Train the transformer model
   - Save the model and associated data

2. **Generating Explanations**:
   ```python
   python ts_explain.py
   ```
   This will:
   - Load the trained model
   - Generate feature importance visualizations
   - Create saliency maps for predictions

## Model Architecture Details

The `SimpleTransformer` class implements a custom transformer architecture with:
- Input feature projection layer
- Positional encoding for temporal information
- Multi-head self-attention mechanism
- Layer normalization
- Feed-forward neural network
- Classification head for binary prediction

## Feature Engineering

The model uses several financial indicators and technical analysis features:
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Simple and Exponential Moving Averages
- Price momentum and volatility measures
- Returns and cumulative returns

## Model Training

The training process includes:
1. Feature selection using L1 regularization
2. Hyperparameter tuning with cross-validation
3. Class-weighted loss function for imbalanced data
4. Early stopping to prevent overfitting
5. Model checkpointing for best performance

## Explainability

The project uses TSInterpret library for model interpretation, providing:
- Feature importance scores
- Temporal saliency maps
- Attribution analysis for predictions

## License

This project is provided as-is for educational and research purposes.
