# MultiWOZ-NLP: Dialogue Action Prediction and Response Generation

This project implements a task-oriented dialogue system using the MultiWOZ dataset version 2.2. The system consists of two main components: **Action Prediction** and **Response Generation**, both built using transformer models (T5) for natural language understanding and generation in multi-domain conversations.

## 🎯 Project Overview

The MultiWOZ-NLP project addresses two critical aspects of task-oriented dialogue systems:

1. **Action Prediction**: Predicting the next dialogue action (intent and slot-value pairs) given the conversation context
2. **Response Generation**: Generating natural language responses based on user input and predicted actions

The system supports multiple domains including hotels, restaurants, attractions, trains, taxis, hospitals, and police services.

## 🏗️ Project Structure

```
multiWOZ-NLP/
├── data/                           # Generated outputs and evaluation results
│   ├── generated_outputs_train_action.pkl
│   ├── generated_outputs_train_response.pkl
│   ├── generated_outputs_valid_action.pkl
│   ├── generated_outputs_valid_response.pkl
│   ├── train_evaluation_response.pkl
│   └── valid_evaluation_response.pkl
├── models/                         # Trained model checkpoints
│   ├── multixoz_action_model.pth
│   └── multixoz_response_model.pth
├── src/                           # Source code directory
│   ├── notebooks/                 # Jupyter notebooks for training and experimentation
│   │   ├── action_model.ipynb     # Action prediction model training
│   │   └── response_model.ipynb   # Response generation model training
│   └── scripts/                   # Python modules and utilities
│       ├── global_vars.py         # Global configuration variables
│       ├── utils.py               # Utility functions for evaluation and plotting
│       ├── preprocessing/         # Data preprocessing modules
│       │   ├── action.py          # Action prediction data processing
│       │   ├── response.py        # Response generation data processing
│       │   └── delexicalization.py # Text delexicalization utilities
│       └── pytorch/               # PyTorch training and inference modules
│           ├── training.py        # Model training functions
│           └── inference.py       # Model inference functions
├── .gitignore                     # Git ignore file
├── .gitattributes                 # Git attributes configuration
├── LICENSE                        # Project license
└── README.md                      # This file
```

## 🔧 Key Components

### 1. Global Configuration (`global_vars.py`)
Contains all project-wide configuration parameters:
- **Model Selection**: T5-small transformer model
- **Training Parameters**: Batch size (256), sequence lengths
- **Device Configuration**: Automatic GPU/MPS/CPU detection
- **Dialogue Context**: Maximum turns for context window

### 2. Data Preprocessing

#### Action Preprocessing (`preprocessing/action.py`)
- **ActionDataset**: Custom dataset class for action prediction
- **Context Management**: Handles multi-turn conversation context
- **Label Processing**: Converts dialogue acts to structured format
- **Evaluation Metrics**: BLEU score and Slot Error Rate (SER)

#### Response Preprocessing (`preprocessing/response.py`)
- **ResponseDataset**: Custom dataset class for response generation
- **Context Integration**: Combines user input with predicted actions
- **Evaluation Metrics**: BLEU score and BERT-F1 score

#### Delexicalization (`preprocessing/delexicalization.py`)
- **Slot Abstraction**: Replaces specific values with generic placeholders
- **Pattern Matching**: Handles slot-value extraction and replacement
- **Text Normalization**: Ensures consistent format across data

### 3. Model Training (`pytorch/training.py`)
- **Training Loop**: Implements epoch-based training with progress tracking
- **Validation**: Includes validation loss computation
- **Optimization**: Uses AdamW optimizer with learning rate scheduling
- **Checkpointing**: Saves best model states based on validation performance

### 4. Model Inference (`pytorch/inference.py`)
- **Batch Processing**: Efficient batch-wise inference
- **Generation Control**: Configurable output length and generation parameters
- **Output Processing**: Token decoding and response formatting

### 5. Utilities (`utils.py`)
- **Performance Visualization**: Plots training/validation metrics
- **Statistical Analysis**: Computes dataset statistics (zero padding percentage)
- **Evaluation Helpers**: Common evaluation functions

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Transformers library
- NLTK
- bert-score
- datasets library

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd multiWOZ-NLP

# Install required packages
pip install torch transformers datasets nltk bert-score matplotlib numpy tqdm
```

### Training Models

#### 1. Action Prediction Model
Open `src/notebooks/action_model.ipynb` and run all cells to:
- Load the MultiWOZ v2.2 dataset
- Preprocess dialogue data for action prediction
- Train the T5 model for dialogue action prediction
- Evaluate model performance using BLEU and SER metrics

#### 2. Response Generation Model
Open `src/notebooks/response_model.ipynb` and run all cells to:
- Load the preprocessed dataset
- Train the T5 model for response generation
- Evaluate using BLEU and BERT-F1 scores
- Generate sample responses

### Model Architecture

Both models use **T5-small** (Text-To-Text Transfer Transformer):
- **Encoder-Decoder Architecture**: Handles variable-length input/output
- **Attention Mechanism**: Captures long-range dependencies in dialogues
- **Pre-trained Weights**: Leverages T5's pre-training for better performance

### Input/Output Formats

#### Action Prediction
- **Input**: Multi-turn conversation context (USER: ... SYS: ...)
- **Output**: Structured dialogue acts (e.g., "inform(name=restaurant_x, area=centre)")

#### Response Generation
- **Input**: User utterance + predicted dialogue action
- **Output**: Natural language system response

## 📊 Evaluation Metrics

### Action Prediction
- **BLEU Score**: Measures n-gram overlap between predicted and ground truth actions
- **Slot Error Rate (SER)**: Measures slot-value prediction accuracy

### Response Generation
- **BLEU Score**: Evaluates lexical similarity
- **BERT-F1**: Measures semantic similarity using BERT embeddings

## 🎮 Usage Examples

### Action Prediction
```python
# Example context
context = "USER: I need a restaurant in the centre SYS: What type of food would you like?"

# Expected output
action = "request(food)"
```

### Response Generation
```python
# Example input
user_input = "I need a restaurant in the centre"
action = "request(food)"

# Expected output
response = "What type of food would you like?"
```

## 🔍 Features

- **Multi-domain Support**: Handles 7 different domains (hotel, restaurant, etc.)
- **Context-aware Processing**: Maintains conversation history for better predictions
- **Flexible Architecture**: Easily extensible for new domains or tasks
- **Comprehensive Evaluation**: Multiple metrics for thorough performance assessment
- **Delexicalization Support**: Optional abstraction of slot values for better generalization

## 📈 Performance

The models achieve competitive performance on the MultiWOZ v2.2 dataset:
- Action prediction models show strong performance in slot error rate
- Response generation models produce fluent and contextually appropriate responses
- Evaluation results are saved in the `data/` directory for analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add appropriate tests
5. Submit a pull request

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🙏 Acknowledgments

- MultiWOZ dataset creators for providing the comprehensive multi-domain dialogue dataset
- Hugging Face Transformers library for transformer model implementations
- T5 model authors for the Text-To-Text Transfer Transformer architecture