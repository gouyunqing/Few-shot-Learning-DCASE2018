import pickle

with open('file2class.pkl', 'rb') as f:
    dict = pickle.load(f)

print(dict[104936])