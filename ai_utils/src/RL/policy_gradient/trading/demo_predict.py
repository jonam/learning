import numpy as np
import torch
from policy_network import PolicyNetwork

def predict_action(model, state):
    """
    Predict the action for a given state using the loaded model.
    
    Parameters:
    - model: The loaded PyTorch model.
    - state: The current state as a NumPy array.
    
    Returns:
    - action: The predicted action.
    """
    # Convert the state to a tensor
    state_tensor = torch.from_numpy(state).float().unsqueeze(0)
    
    # Forward pass to get logits
    with torch.no_grad():
        probs = model(state_tensor)

    probs = torch.softmax(probs, dim=1).numpy().flatten()
 
    return probs
    
    # Convert probabilities to a numpy array
    #probs_np = probs.numpy()
    
    # Select the action with the highest probability
    # action = np.argmax(probs_np)
    #action = probs.numpy().flatten()

    #return action

def normalize_allocations(allocations):
    """
    Normalize the raw model output to ensure allocations sum to 1.
    
    Parameters:
    - allocations: The raw model output as a NumPy array.
    
    Returns:
    - A NumPy array with normalized allocation percentages.
    """
    positive_allocations = np.maximum(allocations, 0)  # Ensure allocations are non-negative
    total_allocation = np.sum(positive_allocations)
    if total_allocation > 0:
        return positive_allocations / total_allocation
    else:
        return np.array([0.5, 0.5])  # Default to even distribution if sum is 0

def softmax_normalization(raw_scores):
    """Apply softmax to raw scores to get normalized, non-negative allocations."""
    softmax_scores = torch.softmax(torch.tensor(raw_scores), dim=0).numpy()
    return softmax_scores

def custom_normalization(raw_scores):
    """Apply a custom normalization to raw scores to get valid allocations."""
    # Ensure all scores are positive
    min_score = min(raw_scores)
    adjusted_scores = [score - min_score for score in raw_scores]
    
    # Normalize to sum to 1
    total_score = sum(adjusted_scores)
    if total_score > 0:
        normalized_scores = [score / total_score for score in adjusted_scores]
    else:
        # Avoid division by zero; equally distribute if all original scores were the same
        normalized_scores = [1.0 / len(raw_scores)] * len(raw_scores)
    
    return normalized_scores

# Example usage
# Assuming `new_state` is a NumPy array representing the state you want to predict the action for
# new_state = np.array([...])
# action = predict_action(policy_loaded, new_state)
# print("Predicted action:", action)


# Assuming the PolicyNetwork class is defined in your script

# Initialize a new instance of the model
policy_loaded = PolicyNetwork()

# Load the state dict back into this model
policy_loaded.load_state_dict(torch.load('policy_model.pth'))

# Make sure to switch to evaluation mode for predictions
policy_loaded.eval()

data = np.array([
    [100, 200],
    [205, 195],
    [110, 190],
    [315, 185],
    [120, 180]
])

# Example usage
# Assuming `new_state` is a NumPy array representing the state you want to predict the action for
for new_state in data:
    #new_state = np.array([105, 210])
    action = predict_action(policy_loaded, new_state)
    normalized_action = normalize_allocations(action)
    print(f"State: {new_state}, Normalized prediction: {normalized_action}")
