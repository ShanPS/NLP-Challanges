''' predict the gender from names '''

# imports
import re


male_pronoun = ['he', 'him', 'his', 'himself']
female_pronoun = ['she', 'her', 'hers', 'herself']

with open('corpus.txt', encoding='utf-8') as corpus:
    data = corpus.read()

with open('input.txt', encoding='utf-8') as input_file:
    n = next(input_file)
    names = input_file.read()

data = re.sub('[^0-9a-zA-Z_\.\'\"\-\n]+', ' ', data)
data = data.lower()
data = data.split('.\n')
names = names.lower()
names = names.split('\n')

predictions = []

for name in names:
    male_pronoun_count = 0
    female_pronoun_count = 0
    for line in data:
        if(' '+name+' ' in line):
            for pronoun in male_pronoun:
                if(pronoun in line):
                    male_pronoun_count += line.count(' '+pronoun+' ')
            for pronoun in female_pronoun:
                if(pronoun in line):
                    female_pronoun_count += line.count(' '+pronoun+' ')
    if(male_pronoun_count > female_pronoun_count):
        predictions.append('male')
    else:
        predictions.append('female')

for pred in predictions:
    print(pred)
