---
# Error-Centric Learning for Reasoning Improvement in GUI Agent (AAAI 2026)
<div align="center">
  <img width="65%" src="IFF-Framework.png">
</div>

## Introduction

**IFF-GUI-3B** is a GUI-tailored foundational model built upon **Qwen2.5-VL-3B-Instruct**. It was developed through a multi-stage training process, incorporating a series of advanced training strategies and a specialized data synthesis pipeline, as detailed in our paper.

The model demonstrates stable short-term memory, exceptional screen comprehension, and robust generalization capabilities. It has achieved state-of-the-art (SOTA) performance on our proprietary Chinese Out-of-Distribution (OOD) GUI automation benchmark.

### Key Features
*   **Excellent Screen Comprehension:** Accurately understands and interprets complex GUI layouts.
*   **Stable Short-Term Memory:** Maintains context and state throughout multi-step tasks.
*   **Robust Generalization:** Performs well on unseen applications and UI designs.
*   **SOTA Performance:** Leads the charts on our Chinese OOD GUI automation benchmark.

## Quick Start

### 1. Download Model Weights

We provide the foundational model and several fine-tuned versions for specific tasks. **IFF-GUI-3B** is the base model, while the others are fine-tuned on top of it.

| Model | Description | Download Link |
| :--- | :--- | :--- |
| **IFF-GUI-3B** | The foundational GUI model. | [Link to be added] |
| **IFF-GUI-3B-AITW** | Fine-tuned on the AITW dataset. | [Link to be added] |
| **IFF-GUI-3B-Mind2Web**| Fine-tuned on the Mind2Web dataset. | [Link to be added] |
| **IFF-GUI-3B-GUIOdyssey**| Fine-tuned on the GUIOdyssey dataset. | [Link to be added] |
| **IFF-GUI-3B-GUI-Understanding**| Fine-tuned for general GUI understanding tasks. | [Link to be added] |

### 2. Environment Setup

We recommend using a virtual environment to manage dependencies.

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
```

## Custom Fine-tuning

We provide a script to fine-tune the model on your own custom datasets or tasks. Before running, you may need to modify the script to point to your data and set your desired training parameters.

```bash
# Run the training script
sh training.sh
```

## Data Synthesis Pipeline

This project includes scripts to replicate our data synthesis process.

*   **Guideline Synthesis:** Generates instruction-following data based on high-level guidelines.
    ```bash
    sh guideline.sh
    ```
*   **System-2 CoT Synthesis:** Creates Chain-of-Thought reasoning data.
    ```bash
    sh CoT.sh
    ```
*   **Action-to-Summary (Act2Sum) Synthesis:** Generates summaries from action sequences to enhance model memory.
    ```bash
    sh Act2Sum.sh
    ```

## Deployment and Inference

We provide three example implementations for model deployment in the `/deployment_examples` directory.

To accelerate inference speed, we support two primary methods:
1.  **Local vLLM Deployment:** For high-throughput, optimized inference.
2.  **Swift Framework:** An alternative efficient deployment framework.

Please refer to the code in our example directories (e.g., `./inference/vllm_example`) for implementation details and adjustments.

## Usage Example: GUI Trajectory Inference

The following Python script demonstrates how to use the model for a GUI automation task. This example assumes you have a local vLLM server running the model. You can adapt the code to fit your specific needs.

```python
import requests
import json

# Define the API endpoint for your local vLLM server
VLLM_URL = "http://localhost:8000/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

# NOTE: Load the appropriate prompt from the `/instruction_templates` directory
# For this example, we assume MIND2WEB_SUM_PROMPT is a loaded string variable.
MIND2WEB_SUM_PROMPT = "Your loaded prompt template here. Goal: (goal), Action: (action)"

def get_model_response(img_url: str, goal: str, action: str):
    """Sends a request to the model and gets a response."""
    
    # Format the prompt with the current task details
    query = MIND2WEB_SUM_PROMPT.replace("(goal)", goal).replace("(action)", action)
    
    # Prepare the payload with image and text
    content = [
        {"type": "image_url", "image_url": {"url": img_url}},
        {"type": "text", "text": query}
    ]
    
    data = {
        # Use the name of the model you deployed, e.g., "IFF-GUI-3B"
        "model": "IFF-GUI-3B", 
        "messages": [
            {"role": "system", "content": "You are a helpful assistant specialized in UI automation."},
            {"role": "user", "content": content}
        ],
        "temperature": 0.0,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(VLLM_URL, headers=HEADERS, data=json.dumps(data))
        response.raise_for_status()  # Raise an exception for bad status codes
        response_json = response.json()
        return response_json['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# --- Example task execution loop ---
# `episode` would be a list of task steps, where each step has an image, goal, and action
# episode = [(img1, goal1, action1), (img2, goal2, action2), ...] 

# Act2Sum_results = []
# for step in episode:
#   img_path, current_goal, performed_action = step
#   act2sum_result = get_model_response(img_path, current_goal, performed_action)
#   if act2sum_result:
#       print(f"Generated Summary: {act2sum_result}")
#       Act2Sum_results.append(act2sum_result)
#   # ... process the result ...

# ... Save and format your collected data ...
```

## Instruction Templates

All instruction templates used in our paper and experiments can be found in the `/instruction_templates` directory. These are crucial for replicating our results and for prompting the model effectively.

## Citation

If you use IFF-GUI-3B or its derivatives in your research, please cite our paper (coming soon) 🙏
