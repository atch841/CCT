import json
import logging
import sys

# logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s %(message)s')

class Logger:
    """
    Training process logger

    Note:
        Used by BaseTrainer to save training history.
    """
    def __init__(self, log_path):
        logging.basicConfig(filename=log_path + 'log.txt',
            level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', 
            datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)
