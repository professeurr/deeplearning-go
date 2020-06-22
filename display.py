import pandas as pd


epochs = 50
file_name = 'metrics_{}.csv'.format(epochs)
data = pd.read_csv(file_name)
data.plot(y=['train_policy_loss', 'val_policy_loss', 'train_policy_accuracy', 'val_policy_accuracy'],
          style=['r', 'b', 'g', 'o'])

