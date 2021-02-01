# load data
output_path = 'outputs\\Heart'
dl = DataLoader('data\\Heart.csv', output_path, 'Heart')
dl.load_data()
dl.scaled_data()
X, y = dl.get_data()