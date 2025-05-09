import torch
import numpy as np
import os

dev = 'cpu' 
Num_X = 25000 

def generate_Q(n):
    Q = 10 * torch.rand(n, n, device=dev, dtype=torch.float) - 5
    return (Q + Q.T) / 2  

def save_Q_to_file(filename):
    Q = generate_Q(Num_X)
    Q_np = Q.cpu().numpy()
    np.save(filename, Q_np) 

if __name__ == "__main__":
    for i in range(5):  
        save_Q_to_file(f"8_data_{Num_X}/data_{i+1}")

