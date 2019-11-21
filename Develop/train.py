from camclassifier import DataLoader

D = DataLoader.DataLoader('blub.flist', (100,100))

pipeline = D.pipeline(32)
print(pipeline)