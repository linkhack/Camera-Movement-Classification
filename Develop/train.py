from camclassifier import DataLoader

D = DataLoader.DataLoader('annotation.flist', (100,100))

pipeline = D.pipeline(32)
print(pipeline)