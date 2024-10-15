# GOT-OCR Wrapper

## Description

Wrapper for the [GOT-OCR2_0](https://huggingface.co/stepfun-ai/GOT-OCR2_0) model providing end-to-end text detection and recognition.  
The wrapper takes a [`VideoDocument`](http://mmif.clams.ai/vocabulary/VideoDocument/v1) with SWT [`TimeFrame`](http://mmif.clams.ai/vocabulary/TimeFrame/v1) annotations and returns [`TextDocument`](http://mmif.clams.ai/vocabulary/TextDocument/v1) [aligned](http://mmif.clams.ai/vocabulary/Alignment/v1) with [`TimePoint`](http://mmif.clams.ai/vocabulary/TimePoint/v1) annotations from the middle frame of each TimeFrame.


![GOT-OCR Framework](https://huggingface.co/stepfun-ai/GOT-OCR2_0/raw/main/framework.jpeg)

## User Instruction

General user instructions for CLAMS apps are available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

### System Requirements

- **Dependencies:**
  - `mmif-python[cv]` for the `VideoDocument` helper functions
- **Hardware:**
  - GPU is required to run the OCR model at a reasonable speed

### Configurable Runtime Parameters

For a comprehensive list of parameters, refer to the app metadata in the [CLAMS App Directory](https://apps.clams.ai) or the [`metadata.py`](metadata.py) file in this repository.

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/clams-project/app-got-ocr-wrapper.git
   cd app-got-ocr-wrapper
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

Here's the framework from the EasyOCR repo:
![EasyOCR Framework](https://github.com/JaidedAI/EasyOCR/raw/master/examples/easyocr_framework.jpeg)

## User instruction

General user instructions for CLAMS apps is available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

### System requirements

- Requires mmif-python[cv] for the `VideoDocument` helper functions
- Requires GPU to run at a reasonable speed

### Configurable runtime parameter

For the full list of parameters, please refer to the app metadata from [CLAMS App Directory](https://apps.clams.ai) or [`metadata.py`](metadata.py) file in this repository.