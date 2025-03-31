import os
import json

SINGLE_FILE_PATH = "single_file_dataset"
MAPPING_PATH = "mappings"
SEQUENCE_LENGTH = 64

class Processor():
    """
    class that offers utilities to process and store raw harmony data
    """
    def __init__(self, raw_data, single_file_path, mapping_path, sequence_length):
        """
        initializes processor class
        """
        self.raw_data = raw_data
        self.single_file_path = single_file_path
        self.mapping_path = mapping_path
        self.sequence_length = sequence_length
        self.filtered_data = self._get_dt_files()
        
    def process(self):
        """
        processes and stores dataset in single file as well as integer mapping 
        dict
        Returns:
            None: nothing
        """
        songs = ""
        for song in self.filtered_data:
            #load and return string
            song_content = self._load_string(song)
            songs = songs + song_content + " "
            #append to single file (inclde new song delimiters)
        songs = songs[:-1]    
        #create mappings from unique tokens in single file
        with open(self.single_file_path, "w") as fp:
            fp.write(songs)
        
        self._generate_mappings(songs)
            
        return None
    
    def _generate_mappings(self, data):
        """
        generates and stores integer mappings for each unique symbol
        as json
        Args:
            data (str): string representation of dataset

        Returns:
            dict: each unique symbol mapped to an integer
        """
        mappings = {}
        
        data = data.split()
        
        vocabulary = list(set(data))

        for i, symbol in enumerate(vocabulary):
            mappings[symbol] = i 
        
        with open(self.mapping_path, "w") as file:
            json.dump(mappings, file, indent=4)
             
        return mappings
    
    def _get_dt_files(self):
        """
        fetches all dt-annotated files
        Returns:
            list: all dt-annotated files
        """
        dt_files = []
        for file in os.listdir(self.raw_data):
            if "dt" in file:
                dt_files.append(file)   
        return dt_files
    
    def _load_string(self, song):
        """
        converts txt file to a single string beginning at first bracket
        to eliminate warnings
        Args:
            song (str): name of song txt file

        Returns:
            str: string representation of txt file
        """
        #return string representation of given txt file 
        with open(self.raw_data + '/' + song, 'r') as file:
                file_content = file.read()
        index = file_content.find('[')
        return file_content[index:]

def main():
    """main function to process and store the data
    """
    #processor class instantiation variables
    raw_data = 'rs200_harmony_exp'
    #processing data
    processor = Processor(raw_data, SINGLE_FILE_PATH, MAPPING_PATH, SEQUENCE_LENGTH)
    processor.process()
    
if __name__ == "__main__":
    main()
