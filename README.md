**Instructions to run**

- create python venv with python 3.10
- install required packages: *$pip install -r requirements.txt*
- preprocessing and training have already been done, but if you wish to re-do them 
      run rc_preprocess.py: *$ python rc_preprocess.py*
      run training module: *$ python train.py*
- generate chord sequenc: *$ python chordgen.py*\

### Completion report ###

**Preprocessing**

- Processed only the dt interpretations
- read raw data, loaded single string in single_file_dataset, removed warnings 
- generated int mappings from unique symbols in dataset

**Training**

- converted data to integer format, generated input, target pairs from given sequence length
- built model using keras functional api approach
- trained and saved model to model.h5 file

**Generation**

- Wrapped generation process in LSTMChordGenerator class
- Model generates probability dist for next symbol
- next symbol sampled using temperature
- Counts amount of chords present in sequence stops at given number (default 20)
- prints generated progression


