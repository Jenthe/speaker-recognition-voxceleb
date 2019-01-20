# speaker-recognition-voxceleb

## Usage

Training a model, optionally starting from a pre-trained model:

    python3 train.py <constants> [model]
  
Testing an individual model or all models in a folder for speaker identification:

    python3 test.py <constants> <model|folder>
  
Testing an individual model or all models in a folder for speaker verification:

    python3 verify.py <constants> <model|folder>
