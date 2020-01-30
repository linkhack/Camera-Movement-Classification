import yaml
from typing import List

stream = open('config.yml', 'r')
config = yaml.load(stream, Loader=yaml.SafeLoader)
print(config)
print(config.get('model'))

def bla(input:List[int]):
    print(input)

