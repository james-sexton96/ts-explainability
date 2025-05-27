from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR
from TSInterpret.InterpretabilityModels.Saliency.SaliencyMethods_PTY import Saliency_PTY
import matplotlib.pyplot as plt
import numpy as np
import torch
from simple_transformer_ts_classification import SimpleTransformer
import pickle

# Load the trained model
def load_model(filepath="final_model.pth"):
    # Load the state dictionary to infer model dimensions
    state_dict = torch.load(filepath)
    # Get model_dim from the first layer of feature_proj (output dimension)
    model_dim = state_dict['feature_proj.0.weight'].shape[0]
    # Get input_dim from the first layer of feature_proj (input dimension)
    input_dim = state_dict['feature_proj.0.weight'].shape[1]

    model = SimpleTransformer(
        input_dim=input_dim,  # Dynamically set input_dim
        model_dim=model_dim,  # Dynamically set model_dim
        num_heads=4,
        num_layers=4,
        num_classes=2,
        dropout=0.3
    )
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {filepath} with model_dim={model_dim}")
    return model

model_to_explain = load_model("final_model.pth")

# Load test data from the data directory
test_data = np.load('data/test_data.npz')
X = test_data['X_test']  # Use test data directly

# Define num_features based on the shape of X
num_features = X.shape[2]

# Load scalers from the "data" folder
scalers_path = "data/scalers.pkl"
with open(scalers_path, "rb") as f:
    scalers = pickle.load(f)

# Scale features using the loaded scalers
for i in range(num_features):
    X[:, :, i] = scalers[i].transform(X[:, :, i])

# Load training data from the "data" folder
train_data_path = "data/train_data.npz"
train_data = np.load(train_data_path)
train_x = train_data['X_train']

# Use the 2025-01-01 to 2025-05-01 data as test_x
test_x = X

# Use TSR to generate explanations
print("Running TSR for explanations...")
# int_mod = TSEvo(model_to_explain, train_x.shape[-2], train_x.shape[-1])
int_mod=Saliency_PTY(model_to_explain, NumTimeSteps=train_x.shape[-2], NumFeatures=train_x.shape[-1], method='FA', mode ='time')

item = np.array([test_x[0, :, :]])
label = 1  # Assuming label 0 for explanation

exp = int_mod.explain(item, labels=1, TSR=True)
print("Explanations generated. Shape:", exp.shape)


# Define feature columns for y-axis labels
feature_cols = [
    'Close', 'Return', 'MA7', 'MA21', 'Volatility7', 'Volatility21',
    'Momentum7', 'Momentum21', 'EMA7', 'EMA21', 'RSI', 'MACD', 'Cumulative_Returns'
]

# Plot the explanations with feature columns as y-axis tick marks
int_mod.plot(
    np.array([test_x[0, :, :]]),
    exp,
    heatmap=False,
    save="explanation_plot.png",
)
# plt.yticks(ticks=np.arange(len(feature_cols)), labels=feature_cols)
# plt.savefig("explanation_plot.png", dpi=300)

print("Explanations plotted successfully.")
