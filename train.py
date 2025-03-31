import json
import keras
import numpy as np

DATA_PATH = "single_file_dataset"
MAPPING_PATH = "mappings"
OUTPUT_UNITS = 158
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 15
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"

def create_training_data(data, sequence_length):
    """_summary_
    creates input and target arrays from mapped integer data
    Args:
        data (list): list of integer-mapped symbols
        sequence_length (int): length of input sequences

    Returns:
        list: input, target pairs
    """
    #split dataset into input and target
    inputs = []
    targets = []

    # generate the training sequences
    num_sequences = len(data) - sequence_length
    for i in range(num_sequences):
        inputs.append(data[i:i+sequence_length])
        targets.append(data[i+sequence_length])

    # one-hot encode the sequences
    vocabulary_size = len(set(data))
    # inputs size: (# of sequences, sequence length, vocabulary size)
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)
    
    print(f"There are {len(inputs)} sequences.")
    
    return inputs, targets

def build_model(output_units, num_units, loss, learning_rate):
    """Builds and compiles model

    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply

    :return model (tf model): Where the magic happens :D
    """

    # create the model architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # compile model
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])

    model.summary()

    return model

def convert_to_int(data):
    """
    converts symbol to its corresponding integer mapping
    Args:
        data (str): path to single file containing training data

    Returns:
        list: data which has been mapped
    """
    with open(data, 'r') as data:
        data = data.read()
        
    int_data = []
    
    # load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # transform songs string to list
    data = data.split()

    # map songs to int
    for symbol in data:
        int_data.append(mappings[symbol])

    return int_data

def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    """
    main function to train and save the model
    Args:
        output_units (int): num output units
        num_units (int): number of lstm units
        loss (str): loss function used
        learning_rate (float): learning rate of the model

    Returns:
        None: nothing
    """
    sequence_length = 64
    int_data = convert_to_int(DATA_PATH)
    
    inputs, targets = create_training_data(int_data, sequence_length)
    # build the network
    model = build_model(output_units, num_units, loss, learning_rate)

    # train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(SAVE_MODEL_PATH)
    return None

if __name__ == '__main__':
    train()
    