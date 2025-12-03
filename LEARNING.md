# Project Learning Summary

## Project Overview
This project is a low-level implementation of a Long Short-Term Memory (LSTM) neural network built entirely from scratch using Python and NumPy. The primary objective was to understand the inner workings of Recurrent Neural Networks (RNNs) by implementing the forward and backward propagation algorithms manually, without relying on high-level deep learning frameworks like TensorFlow or PyTorch. The model is trained on a dataset of French names to perform character-level text generation. Additionally, the project includes a unique feature to export the trained model's weights into `.coe` (Coefficient) files, facilitating deployment onto Xilinx FPGAs.

## Tech Stack and Key Technologies
- **Language:** Python 3.x
- **Core Library:** NumPy (for matrix operations and linear algebra)
- **Data Handling:** Pandas (for CSV manipulation)
- **Visualization:** Matplotlib (for plotting loss, perplexity, and accuracy)
- **Hardware Integration:** Xilinx FPGA (via `.coe` file generation)

## Notable Libraries
- **NumPy:** The backbone of the project. It was used to implement all vector and matrix operations, including the sigmoid and tanh activation functions, gate calculations (Forget, Input, Output, Candidate), and the Adam optimizer.
- **Pandas:** Used for efficient loading and preprocessing of the `FrenchNames.csv` dataset, including one-hot encoding preparation.
- **Matplotlib:** Essential for visualizing the training progress, allowing for the monitoring of loss convergence and accuracy improvements over epochs.

## Major Achievements and Skills Demonstrated
- **Deep Learning from Scratch:**
    - Implemented the complex architecture of an LSTM cell (Forget, Input, Output, and Candidate gates) using only raw matrix math.
    - Manually derived and implemented Backpropagation Through Time (BPTT) to calculate gradients for all weights.
- **Optimization Algorithms:**
    - Implemented the **Adam Optimizer** manually, including the calculation of first and second moments (momentum and RMSprop components) to adapt learning rates.
- **Data Preprocessing:**
    - Developed a pipeline to convert text data into one-hot encoded vectors and handle variable-length sequences via padding.
- **Hardware-Software Co-Design:**
    - Created a bridge between software training and hardware inference by implementing a function to serialize floating-point weights into binary `.coe` format for FPGA initialization.
- **Model Evaluation:**
    - Implemented custom metrics for Loss (Cross-Entropy), Perplexity, and Accuracy to evaluate the generative model's performance.

## Skills Gained/Reinforced
- **Neural Network Fundamentals:** Deep understanding of vanishing gradients, gating mechanisms, and the flow of gradients in recurrent architectures.
- **Numerical Computing:** Proficiency in vectorization and broadcasting with NumPy to optimize performance.
- **Algorithm Implementation:** Ability to translate mathematical formulas (derivatives, update rules) directly into code.
- **FPGA Workflow:** Understanding of how software-defined parameters need to be formatted for hardware description languages or memory initialization.
