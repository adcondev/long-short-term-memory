# LEARNING.md

## Project Overview

This project involves the development of a character-level Long Short-Term Memory (LSTM) network from scratch using Python and NumPy. The primary objective is to generate French names automatically. The model is trained on a dataset of French names and learns the underlying patterns and structures of the language to produce new, plausible-sounding names. A key feature of this project is the functionality to export the trained model weights into a format suitable for deployment on a Field-Programmable Gate Array (FPGA), indicating an application in hardware-accelerated machine learning.

## Tech Stack and Key Technologies

- **Programming Language:** Python
- **Core Libraries:** NumPy, Pandas
- **Data Visualization:** Matplotlib
- **Target Hardware:** FPGA (Field-Programmable Gate Array)

## Notable Libraries

- **NumPy:** Utilized for all numerical operations, including the implementation of the LSTM architecture, activation functions, and the Adam optimization algorithm from scratch. This demonstrates a strong understanding of the mathematical foundations of deep learning.
- **Pandas:** Employed for data loading and initial preprocessing of the French names dataset from a CSV file.
- **Matplotlib:** Used for visualizing the model's performance metrics, such as loss, perplexity, and accuracy, over the training epochs.

## Major Achievements and Skills Demonstrated

- **Implemented a Character-Level LSTM from Scratch:** Designed and built a complete LSTM network, including the forward and backward propagation steps, without relying on high-level deep learning frameworks like TensorFlow or PyTorch.
- **Developed Custom Activation and Optimization Functions:** Coded sigmoid, tanh, and softmax activation functions, as well as their derivatives, and implemented the Adam optimization algorithm manually.
- **Engineered a Data Preprocessing Pipeline:** Created a pipeline to load, clean, and transform the text data into a suitable format (one-hot encoding) for training the character-level LSTM.
- **Created a Weight Export Mechanism for FPGAs:** Developed a specific function to convert and save the trained model weights into a binary format (`.coe` files), preparing them for deployment on hardware.
- **Conducted Model Training and Evaluation:** Successfully trained the LSTM model and evaluated its performance using metrics like loss, perplexity, and accuracy, including visualizations of the results.

## Skills Gained/Reinforced

- **Deep Learning Fundamentals:** In-depth understanding and practical application of recurrent neural networks (RNNs), specifically LSTM architecture.
- **Numerical Computing:** Advanced proficiency in NumPy for scientific and numerical computing.
- **Algorithm Implementation:** Experience in implementing complex algorithms like backpropagation through time (BPTT) and Adam optimization.
- **Software Engineering:** Skills in structuring a machine learning project, including data handling, model implementation, training, and evaluation.
- **Hardware-Software Co-design:** Exposure to the process of preparing a machine learning model for hardware acceleration on FPGAs.
- **Data Visualization:** Competence in using Matplotlib to create informative plots for model analysis.
