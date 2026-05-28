# Neural Network from Scratch

A feedforward neural network implemented using only NumPy to classify points in a 3D hat-shaped region.

## Results
- **Accuracy:** 98.1%
- **Final Loss:** 0.017

## Network Architecture
- Input: 3 neurons (x, y, z)
- Hidden layers: 3 layers with 10 neurons each
- Output: 1 neuron (binary classification)

## Run the Code
```bash
pip install numpy matplotlib
python hat_detector.py
```
## How it Works
- Forward propagation through 4 layers
- Calculate loss (Mean Squared Error)
- Backpropagation to compute gradients
- Update weights with gradient descent

Author
mahdieh_gh
