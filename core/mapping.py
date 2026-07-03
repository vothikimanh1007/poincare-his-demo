import numpy as np

def poincare_distance(u, v, c=1.0):
    """
    Calculates the geodesic distance between two points in the Poincaré ball.
    Formula: d_c(x, y) = arcosh(1 + 2 * ||x-y||^2 / ((1 - c||x||^2)(1 - c||y||^2)))
    """
    sq_dist = np.sum((u - v) ** 2)
    u_norm_sq = np.sum(u ** 2)
    v_norm_sq = np.sum(v ** 2)
    
    # Epsilon to prevent division by zero near the boundary horizon
    eps = 1e-5 
    denom = (1 - c * u_norm_sq) * (1 - c * v_norm_sq) + eps
    
    # Ensure value inside arccosh is >= 1 to prevent NaN errors
    arg = 1 + 2 * c * sq_dist / denom
    arg = np.clip(arg, 1.0 + eps, None) 
    
    return np.arccosh(arg) / np.sqrt(c)

def apply_scaling_brake(vector, alpha=0.1):
    """
    Paper Contribution #1: The Scaling Brake.
    Prevents boundary mode collapse by scaling down Euclidean logits 
    before they are projected via the exponential map.
    """
    return vector * alpha

def mock_encode_and_project(text_data, image_data):
    """
    Simulates the ClinicalBERT (text) and ResNet (vision) encoders, 
    applies the scaling brake, and computes the Möbius addition 
    to return a fused 3D coordinate in the Poincaré ball.
    """
    # 1. Simulate high-dimensional Euclidean embedding from Encoders
    raw_vector = np.random.randn(3) 
    
    # 2. Apply Scaling Brake (alpha = 0.1) as defined in the paper
    braked_vector = apply_scaling_brake(raw_vector, alpha=0.1)
    
    # 3. Ensure representation is strictly inside the open unit ball (r < 1)
    norm = np.linalg.norm(braked_vector)
    if norm >= 1.0:
        braked_vector = braked_vector / (norm + 1e-5)
        
    return braked_vector
