
class DataLoader:
    def __init__(self, filepath: str | None = None, dataset: str | None = None):
        self.filepath = filepath
        self.dataset = dataset

    def load_file(self):
        with open(self.filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return lines