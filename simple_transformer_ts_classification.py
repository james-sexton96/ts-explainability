import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import json
import os
import pickle

# Feature engineering

# Add new features
# RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Move the prepare_data function to the top of the file
def prepare_data():
    # Download AAPL daily stock data for 2024
    data = yf.download('AAPL', start='2005-01-01', end='2024-12-31')

    # Use closing price for simplicity
    data = data[['Close']]
    data['Return'] = data['Close'].pct_change()
    data['MA7'] = data['Close'].rolling(window=7).mean()
    data['MA21'] = data['Close'].rolling(window=21).mean()
    data['Volatility7'] = data['Close'].rolling(window=7).std()
    data['Volatility21'] = data['Close'].rolling(window=21).std()
    data['Momentum7'] = data['Close'] - data['Close'].shift(7)
    data['Momentum21'] = data['Close'] - data['Close'].shift(21)
    data['EMA7'] = data['Close'].ewm(span=7, adjust=False).mean()
    data['EMA21'] = data['Close'].ewm(span=21, adjust=False).mean()

    # Add new features
    data['RSI'] = calculate_rsi(data)
    data['MACD'] = data['EMA7'] - data['EMA21']

    if 'High' in data.columns and 'Low' in data.columns:
        data['High_Low_Range'] = data['High'] - data['Low']
    else:
        data['High_Low_Range'] = 0

    data['Cumulative_Returns'] = (1 + data['Return']).cumprod()
    data = data.dropna()

    return data

# Simple Transformer Model
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout=0.2):
        super().__init__()
        # Initial feature projection with batch normalization
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding for better temporal understanding
        self.pos_encoder = nn.Parameter(torch.randn(1, 21, model_dim))  # 21 is seq_length
        
        # Transformer layers with layer normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim*4,  # Increased feedforward dimension
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Enhanced classification head
        self.classification_head = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.BatchNorm1d(model_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes)
        )
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Project features
        x = x.view(-1, x.size(-1))  # Flatten for batch norm
        x = self.feature_proj(x)
        x = x.view(batch_size, seq_len, -1)  # Restore sequence dimension
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)
        
        # Classification
        x = self.classification_head(x)
        return x


if __name__ == "__main__":
    # Ensure data is prepared before accessing it
    data = prepare_data()

    # MACD (Moving Average Convergence Divergence)
    data['MACD'] = data['EMA7'] - data['EMA21']

    # Daily High-Low Range (Assuming High and Low columns exist)
    if 'High' in data.columns and 'Low' in data.columns:
        data['High_Low_Range'] = data['High'] - data['Low']
    else:
        data['High_Low_Range'] = 0

    # Cumulative Returns
    data['Cumulative_Returns'] = (1 + data['Return']).cumprod()

    data = data.dropna()

    # Create a binary classification target: 1 if next day's close is higher, else 0
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data = data.dropna()

    # Prepare sequences for time series classification
    def create_sequences(data, seq_length=10):
        xs, ys = [], []
        feature_cols = [
            'Close', 'Return', 'MA7', 'MA21', 'Volatility7', 'Volatility21',
            'Momentum7', 'Momentum21', 'EMA7', 'EMA21', 'RSI', 'MACD', 'Cumulative_Returns'
        ]
        for i in range(len(data) - seq_length):
            # Create a sliding window of size seq_length
            x = data.iloc[i:i+seq_length][feature_cols].values
            y = data.iloc[i+seq_length]['Target']  # Target corresponds to the next day
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    # Prepare sequences and flatten targets
    seq_length = 21  # Updated sequence length to 21 days
    X, y = create_sequences(data, seq_length)  # Adjusted sequence length
    y = y.flatten()
    print(f"First 5 y values: {y[:5]}\n")

    # Debug: print shapes after sequence creation
    print(f"X shape after create_sequences: {X.shape}")
    print(f"y shape after create_sequences: {y.shape}")

    
    # Since we're using windows, we need to adjust the split point to account for the sequence length
    # to ensure test set windows only contain test period data
    test_size = 1096
    split_idx = len(X) - test_size
    
    # Split the data
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    print(f"Test set represents trading days from approximately last 6 months")

    # Scale features

    # Scale each feature independently
    num_features = X_train.shape[2]
    scalers = [StandardScaler() for _ in range(num_features)]
    for i in range(num_features):
        X_train[:,:,i] = scalers[i].fit_transform(X_train[:,:,i])
        X_test[:,:,i] = scalers[i].transform(X_test[:,:,i])

    # Debug: print shapes after scaling
    print(f"X_train shape after scaling: {X_train.shape}")
    print(f"X_test shape after scaling: {X_test.shape}")

    # Perform feature selection using L1 regularization
    def select_features_with_l1(X_train, y_train, X_test, num_features=10):
        # Get the original shapes
        batch_size, seq_length, n_features = X_train.shape
        
        # Flatten the sequences for feature selection while preserving time steps
        X_train_reshaped = X_train.reshape(-1, n_features)  # Combine batch and sequence dimensions
        X_test_reshaped = X_test.reshape(-1, n_features)
        
        # Replicate the labels for each time step
        y_train_repeated = np.repeat(y_train, seq_length)
        
        # Train a logistic regression model with L1 regularization
        l1_model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42)
        l1_model.fit(X_train_reshaped, y_train_repeated)
        
        # Get the absolute values of coefficients and select top features
        feature_importances = np.abs(l1_model.coef_).flatten()
        top_features_idx = np.argsort(feature_importances)[-num_features:]
        
        # Select the top features while maintaining the 3D structure
        X_train_selected = X_train[:, :, top_features_idx]
        X_test_selected = X_test[:, :, top_features_idx]
        
        # Get feature names for the selected indices
        feature_cols = [
            'Close', 'Return', 'MA7', 'MA21', 'Volatility7', 'Volatility21',
            'Momentum7', 'Momentum21', 'EMA7', 'EMA21', 'RSI', 'MACD', 'Cumulative_Returns'
        ]
        selected_features = [feature_cols[i] for i in top_features_idx]
        print(f"Selected feature indices: {top_features_idx}")
        print(f"Selected features: {selected_features}")
        print(f"Selected features shape - Train: {X_train_selected.shape}, Test: {X_test_selected.shape}")
        return X_train_selected, X_test_selected

    # Apply feature selection after scaling
    X_train, X_test = select_features_with_l1(X_train, y_train, X_test, num_features=10)

    # Update the input_dim for the model - use the number of features (last dimension)
    input_dim = X_train.shape[2]

    # Debug: print shapes after feature selection
    print(f"X_train shape after feature selection: {X_train.shape}")
    print(f"X_test shape after feature selection: {X_test.shape}")
    print(f"Using input_dim={input_dim} (number of features)")

    # PyTorch Dataset
    class StockDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_ds = StockDataset(X_train, y_train)
    test_ds = StockDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)


    # Model, loss, optimizer
    model = SimpleTransformer(input_dim=input_dim, model_dim=64, num_heads=4, num_layers=4, num_classes=2, dropout=0.3)
    # Calculate class weights
    class_counts = np.bincount(y_train.astype(int))
    total_samples = len(y_train)
    class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) for count in class_counts])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # Added weight_decay

    def train(model, loader, criterion, optimizer, epochs=100):
        model.train()
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        for epoch in range(epochs):
            total_loss = 0
            all_labels = []
            all_preds = []
            all_probs = []
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                out = model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                # Collect predictions for metrics
                probs = torch.softmax(out, dim=1)[:,1].detach().cpu().numpy()
                preds = out.argmax(dim=1).detach().cpu().numpy()
                labels = y_batch.detach().cpu().numpy()
                all_labels.extend(labels)
                all_preds.extend(preds)
                all_probs.extend(probs)
            avg_loss = total_loss/len(loader)
            acc = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, zero_division=0)
            rec = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except Exception:
                auc = float('nan')
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
            # Early stopping
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    def evaluate(model, loader, return_metrics=False):
        model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                out = model(X_batch)
                probs = torch.softmax(out, dim=1)[:,1].cpu().numpy()
                preds = out.argmax(dim=1).cpu().numpy()
                labels = y_batch.cpu().numpy()
                all_labels.extend(labels)
                all_preds.extend(preds)
                all_probs.extend(probs)
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = float('nan')
        print(f"Test set metrics -> Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

        if return_metrics:
            return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc}

    # Ensure save_metrics is defined before usage
    def save_metrics(metrics, filename="metrics.json"):
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=4)

    # Ensure param_grid and hyperparameter_tuning are defined before usage
    param_grid = {
        'lr': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        'model_dim': [32, 64, 128, 256],
        'dropout': [0.1, 0.2, 0.3, 0.4],
        'num_layers': [2, 4, 6],
        'num_heads': [4, 8],
        'batch_size': [16, 32, 64]
    }

    def hyperparameter_tuning(model_class, train_loader, test_loader, param_grid, epochs=50):
        best_params = None
        best_f1 = 0
        results = []

        for params in ParameterGrid(param_grid):
            print(f"Testing params: {params}")
            model = model_class(
                input_dim=X_train.shape[2],  # Use number of features as input dimension
                model_dim=params['model_dim'],
                num_heads=4,
                num_layers=4,
                num_classes=2,
                dropout=params['dropout']
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
            train(model, train_loader, criterion, optimizer, epochs=epochs)
            metrics = evaluate(model, test_loader, return_metrics=True)

            results.append({**params, **metrics})
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_params = params

        print(f"Best params: {best_params} with F1: {best_f1:.3f}")
        return best_params, results

    # Check if tuning_results.json exists
    if os.path.exists("tuning_results.json"):
        print("Tuning results found. Skipping hyperparameter tuning.")
        with open("tuning_results.json", "r") as f:
            tuning_results = json.load(f)
            best_params = tuning_results[0]  # Assuming the best params are the first entry
    else:
        # Run hyperparameter tuning
        best_params, tuning_results = hyperparameter_tuning(SimpleTransformer, train_loader, test_loader, param_grid)
        # Save tuning results
        save_metrics(tuning_results, "tuning_results.json")

    # Train the model with the best parameters
    final_model = SimpleTransformer(
        input_dim=X_train.shape[2],  # Use number of features as input dimension
        model_dim=best_params['model_dim'],
        num_heads=4,
        num_layers=4,
        num_classes=2,
        dropout=best_params['dropout']
    )
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=1e-4)

    # Ensure the "data" folder exists
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)

    # Save training and test data to the "data" folder
    train_data_path = os.path.join(data_folder, "train_data.npz")
    test_data_path = os.path.join(data_folder, "test_data.npz")
    np.savez(train_data_path, X_train=X_train, y_train=y_train)
    np.savez(test_data_path, X_test=X_test, y_test=y_test)
    print(f"Training data saved to {train_data_path}")
    print(f"Test data saved to {test_data_path}")

    # Save scalers to the "data" folder
    scalers_path = os.path.join(data_folder, "scalers.pkl")
    with open(scalers_path, "wb") as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved to {scalers_path}")

    # Train the final model
    train(final_model, train_loader, criterion, final_optimizer, epochs=100)

    # Save the trained model
    def save_model(model, filepath="final_model.pth"):
        torch.save(model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    save_model(final_model)

    # Load the model for later use
    def load_model(filepath="final_model.pth"):
        model = SimpleTransformer(
            input_dim=X_train.shape[2],  # Use number of features as input dimension
            model_dim=best_params['model_dim'],
            num_heads=4,
            num_layers=4,
            num_classes=2,
            dropout=best_params['dropout']
        )
        model.load_state_dict(torch.load(filepath))
        model.eval()
        print(f"Model loaded from {filepath}")
        return model

    # Example of loading the model later
    # loaded_model = load_model("final_model.pth")
