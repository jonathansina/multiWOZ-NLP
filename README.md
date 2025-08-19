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
The global variables file serves as the central configuration hub for the entire project, controlling all critical aspects of model behavior, training, and inference. Understanding these variables is crucial for effective use of the system:

#### Model Configuration
```python
MODEL_NAME = "google-t5/t5-small"  # Can be changed to "google/t5-efficient-mini" for faster training
```
- **Purpose**: Defines which pre-trained T5 model to use as the base
- **Impact**: Larger models (like t5-base) provide better performance but require more computational resources
- **Usage**: Simply change this variable to experiment with different model sizes

#### Training Parameters
```python
BATCH_SIZE = 256                    # Number of samples processed together
MAX_LENGTH_ENCODER_ACTION = 64      # Input sequence length for action prediction
MAX_LENGTH_DECODER_ACTION = 32      # Output sequence length for action prediction
MAX_LENGTH_ENCODER_RESPONSE = 64    # Input sequence length for response generation
MAX_LENGTH_DECODER_RESPONSE = 32    # Output sequence length for response generation
MAX_TURNS = 2                       # Number of conversation turns to include as context
```
- **BATCH_SIZE**: Higher values speed up training but require more GPU memory
- **Sequence Lengths**: Balance between context preservation and computational efficiency
- **MAX_TURNS**: Controls how much conversation history is considered for predictions

#### Checkpoint Management
```python
USE_SAVE_CHECKPOINT = False
```
- **Critical Feature**: This boolean flag controls whether to save model checkpoints during training
- **When True**: 
  - Models are automatically saved after training
  - Saved to `models/multixoz_action_model.pth` and `models/multixoz_response_model.pth`
  - Enables loading pre-trained models for inference without retraining
- **When False**: 
  - Models exist only in memory during the session
  - No persistent storage of trained weights
  - Requires retraining for each new session

**💡 Pro Tip**: Set `USE_SAVE_CHECKPOINT = True` for production use or when you want to:
- Skip training and directly use pre-trained models
- Resume training from a previous checkpoint
- Share trained models with team members
- Conduct inference-only experiments

#### Device Configuration
```python
# Automatic device detection with priority: CUDA > MPS > CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")      # NVIDIA GPU acceleration
elif torch.mps.is_available():
    DEVICE = torch.device("mps")       # Apple Silicon acceleration
else:
    DEVICE = torch.device("cpu")       # Fallback to CPU
```
- **Smart Detection**: Automatically selects the best available hardware
- **Performance Impact**: GPU acceleration can be 10-100x faster than CPU
- **Cross-Platform**: Supports NVIDIA GPUs, Apple Silicon, and CPU fallback

#### How to Use These Variables Effectively

1. **Quick Experimentation**: Set `USE_SAVE_CHECKPOINT = False` and reduce `BATCH_SIZE` for rapid prototyping
2. **Production Training**: Set `USE_SAVE_CHECKPOINT = True` with optimal batch size for your hardware
3. **Inference Only**: With saved checkpoints, you can load models directly without any training
4. **Memory Management**: Adjust sequence lengths and batch size based on available GPU memory
5. **Context Control**: Modify `MAX_TURNS` to experiment with different amounts of conversation history

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

#### Understanding Checkpoint Configuration
Before training, it's important to understand the checkpoint system controlled by the `USE_SAVE_CHECKPOINT` variable in `global_vars.py`:

**Option 1: Training with Checkpoints (Recommended)**
```python
USE_SAVE_CHECKPOINT = True  # Set this in global_vars.py
```
- ✅ Models are saved automatically after training
- ✅ Can resume inference without retraining
- ✅ Models persist between sessions
- ✅ Enables model sharing and deployment

**Option 2: Training without Checkpoints (Development)**
```python
USE_SAVE_CHECKPOINT = False  # Set this in global_vars.py
```
- ⚠️ Models exist only during the current session
- ⚠️ Requires retraining for each new experiment
- ✅ Useful for quick experimentation and debugging

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

### Using Pre-trained Models (No Training Required)

If you have saved model checkpoints in the `models/` directory, you can skip training and directly use the models for inference:

#### Loading Saved Models
```python
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from scripts.global_vars import MODEL_NAME, DEVICE

# Load tokenizer and model architecture
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=True)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Load trained weights from checkpoint
model.load_state_dict(torch.load('models/multixoz_action_model.pth', map_location=DEVICE))
model.to(DEVICE)

# Now ready for inference without any training!
```

#### Benefits of Using Checkpoints
1. **Instant Deployment**: No waiting for training completion
2. **Reproducible Results**: Same model weights produce consistent outputs
3. **Resource Efficiency**: Save computational time and energy
4. **Production Ready**: Trained models can be immediately deployed
5. **Experimentation**: Focus on inference and evaluation rather than training

#### Workflow Recommendations
- **First Time**: Set `USE_SAVE_CHECKPOINT = True` and train both models
- **Subsequent Runs**: Load checkpoints and skip directly to inference/evaluation
- **Model Updates**: Only retrain when you modify architecture or training data

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

## 📋 Configuration Examples

### Example 1: Quick Development Setup
```python
# In global_vars.py - for rapid experimentation
MODEL_NAME = "google/t5-efficient-mini"  # Smaller, faster model
BATCH_SIZE = 64                          # Reduced for limited GPU memory
USE_SAVE_CHECKPOINT = False              # No persistence needed
MAX_TURNS = 1                           # Simplified context
```
**Use Case**: Initial development, debugging, quick prototyping

### Example 2: Production Training Setup
```python
# In global_vars.py - for final model training
MODEL_NAME = "google-t5/t5-small"       # Standard model size
BATCH_SIZE = 256                        # Full batch size for efficiency
USE_SAVE_CHECKPOINT = True              # Save for deployment
MAX_TURNS = 2                          # Full context window
```
**Use Case**: Final model training, deployment preparation

### Example 3: Inference-Only Setup
```python
# In global_vars.py - when using pre-trained models
MODEL_NAME = "google-t5/t5-small"       # Must match training configuration
BATCH_SIZE = 512                        # Can be larger for inference
USE_SAVE_CHECKPOINT = True              # Load existing checkpoints
```
**Use Case**: Production inference, model evaluation, demonstration

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