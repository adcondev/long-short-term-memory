# Character-Level LSTM for French Name Generation


[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](PLACEHOLDER_URL)

This project is a character-level Long Short-Term Memory (LSTM) network built from scratch in Python using NumPy. It's designed to generate French names by learning the patterns from a given dataset. The project also includes functionality to export the trained model weights for deployment on an FPGA.

## Architecture Diagram

```mermaid
graph TD;
    A[French Names CSV] --> B[Data Preprocessing in Pandas];
    B --> C[Character Vocabulary & One-Hot Encoding];
    C --> D{LSTM Model Training (NumPy)};
    D --> E[Generate New Names];
    D --> F[Export Weights to .coe Files for FPGA];
```

## Features

- **LSTM from Scratch:** The LSTM model is implemented using only NumPy, providing a deep understanding of the underlying mechanics.
- **Character-Level Generation:** The model learns to generate text one character at a time.
- **FPGA Weight Export:** Includes a utility to convert and save model weights for hardware acceleration.
- **Performance Visualization:** Uses Matplotlib to plot training metrics like loss, perplexity, and accuracy.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repository.git
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created with the following content:)*
    ```
    numpy
    pandas
    matplotlib
    ```

## Usage

To train the model and generate names, run the main script:
```bash
python LSTM.py
```
The script will train the model, display performance plots, and save the exported weights as `.coe` files in the root directory.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please feel free to open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
