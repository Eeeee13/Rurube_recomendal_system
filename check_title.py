import pandas as pd
import numpy as np
from spacy.lang.ru import Russian
from tqdm import tqdm
import torch
from transformers import BertTokenizer
import transformers

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv('labeled.csv')
df.info()
df['toxic'].value_counts()
df.isnull().sum()
df.head(10)
## Предобработка текста
# загружаем предобученную модель для английского языка
nlp = Russian()
# Лемматизируем текста
new_corpus = []

for doc in tqdm(nlp.pipe(df['comment'], batch_size=64, n_process=-1, disable=["parser", "ner"]), total=len(df['comment'])):
    word_list = [tok.lemma_ for tok in doc]
    new_corpus.append(' '.join(word_list))

df['lemm_spacy_new'] = new_corpus  
# Объявляем токенайзер и модель для создания эмбедингов
tokenizer = BertTokenizer.from_pretrained("unitary/toxic-bert")

model = transformers.BertModel.from_pretrained("unitary/toxic-bert")    
tqdm.pandas()
tokenized = df['lemm_spacy_new'].apply(lambda x: tokenizer.encode(x, max_length=128, truncation=True, add_special_tokens=True)) #обрежет под нужное кол-во токенов

padded = pad_sequence([torch.as_tensor(seq) for seq in tokenized], batch_first=True) #добьет нулями  

attention_mask = padded > 0
attention_mask = attention_mask.type(torch.LongTensor) 
dataset = TensorDataset(attention_mask, padded)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0) # создание загрузчика данных с батчом 32

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") #проверяет доступно ли нам GPU или нет
print(f'Device: {device}')
embeddings = []
model.to(device)
model.eval()
for attention_mask, padded in tqdm(dataloader):
    attention_mask, padded = attention_mask.to(device), padded.to(device)

    with torch.no_grad():
        batch_embeddings = model(padded, attention_mask=attention_mask)

    embeddings.append(batch_embeddings[0][:,0,:].cpu().numpy())

features = np.concatenate(embeddings)
## Обучение модели
X_train, X_test, y_train, y_test = train_test_split(features, df['toxic'], test_size=0.001, random_state=322)

# Обучение логистической регрессии
clf = LogisticRegression(max_iter=3000)
clf.fit(X_train, y_train)

def Check_title(df_full):
    df = df_full['title']
    df.tolist()
    res = clf.predict(df)
    return res