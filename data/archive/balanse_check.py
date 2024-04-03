import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

filepath = input()
df = pd.read_csv(filepath)
labels = list(df['label'])
label_counts = Counter(labels)
df = pd.DataFrame.from_dict(label_counts, orient='index')
df.plot(kind='bar')
plt.show()