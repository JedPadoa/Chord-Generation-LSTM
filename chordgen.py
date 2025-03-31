import keras
import numpy as np
import json

NUM_CHORDS = 20
MODEL = "model.h5"
MAPPINGS_PATH = "mappings"

class LSTMChordGenerator():
    """
    class that can generate chord progressions using a pretrained model
    """
    def __init__(self, num_chords, model_path, mappings_path, sequence_length):
        """
        Initiates LSTMChordGenerator object
        Args:
            num_chords (int): number of chords to be generated
            model_path (str): path to the trained model
            mappings_path (str): path to json mappings file
            sequence_length (int): max length of sequence used to make a prediction
        """
        self.num_chords = num_chords
        self.model_path = model_path
        self.mappings_path = mappings_path
        self.model = keras.models.load_model(model_path)
        self.sequence_length = sequence_length
        
        with open(mappings_path, "r") as fp:
            self._mappings = json.load(fp)
    
    def generate(self, seed, num_steps, temperature):
        """
        generates a chord progression by making predictions using the trained 
        model
        Args:
            seed (str): initial seed data
            num_steps (int): maximum number of predictions that will be made
            temperature (float): temperature used to influence sampling

        Returns:
            str: the generated chord progression in string form
        """
        #generate a chord progression of length num_chords
        chord_counter = 0
        
        seed = seed.split()
        
        for symbol in seed:
            if self._isChord(symbol):
                chord_counter += 1
        
        prog = seed
        
        #seed = self._start_symbols + seed
        
        seed = [self._mappings[symbol] for symbol in seed]
        
        for _ in range(num_steps):

            # limit the seed to max_sequence_length
            seed = seed[-self.sequence_length:]
            

            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # (1, max_sequence_length, num of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]
        
            if self._isChord(output_symbol):
                chord_counter += 1
                
            # update melody
            prog.append(output_symbol)
            
            if chord_counter == 20:
                break

        return ' '.join(prog)
    
    def _isChord(self, input):
        """
        Check if input is a chord or a different symbol by 
        analyzing characters
        Args:
            input (str): symbol to be checked
        Returns:
            boolean: true if chord, false if not
        """
        #check if given input is a chord
        chars = ['I', 'V', 'i', 'v']
        if any(char in input for char in chars):
            return True   
        return False
    
    def _sample_with_temperature(self, probabilites, temperature):
        """Samples an index from a probability array reapplying softmax using temperature

        :param predictions (nd.array): Array containing probabilities for each of the possible outputs.
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.

        :return index (int): Selected output symbol
        """
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index

def generate_chord_progression():
    """main method to generate a chord progression using the model
    """
    seed = 'I | ii7 | V | I'
    generator = LSTMChordGenerator(NUM_CHORDS, MODEL, MAPPINGS_PATH, 64)
    prog = generator.generate(seed, 500, 0.7)
    print(f"generated chord progression of length {NUM_CHORDS}: " + prog)

if __name__ == "__main__":
    generate_chord_progression()