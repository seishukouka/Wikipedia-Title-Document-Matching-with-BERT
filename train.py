import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import json
import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

logging.disable(logging.WARNING)

def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Tohoku UniversityのBERTモデルとトークナイザーの初期化
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')


# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# JSONファイルの読み込み
train_data = load_data('train_q.json')
test_data = load_data('test_q.json')
exam_data = load_data('exam_data1.json')

# テキストデータの準備
texts = {item[0]: item[1] for item in exam_data}

class TitleDocDataset(Dataset):
    def __init__(self, dataframe, tokenizer, texts, max_length=512):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title_id = self.data.iloc[idx]['title_id']
        doc_id = self.data.iloc[idx]['doc_id']
        label = self.data.iloc[idx]['label'] 
        title_text = self.texts[title_id]
        doc_text = self.texts[doc_id]
        inputs = self.tokenizer.encode_plus(
            title_text,
            doc_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            truncation_strategy='only_first'
        )
        return {
            'ids': inputs['input_ids'].flatten(),
            'mask': inputs['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long)
        }

class TitleDocClassifier(nn.Module):
    def __init__(self, bert_model):
        super(TitleDocClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, ids, mask):
        _, pooled_output = self.bert(ids, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        output = self.out(dropout_output)  
        return output

train_df = pd.DataFrame(columns=['title_id', 'doc_id', 'label'])
rows = []

for item in train_data:
    title_id = item['title_id']
    ans_id = item['ans_id']
    for doc_id in item['candidates']:
        label = 1 if doc_id == ans_id else 0
        rows.append({'title_id': title_id, 'doc_id': doc_id, 'label': label})

train_df = pd.concat([train_df, pd.DataFrame(rows)], ignore_index=True)


# データセットのインスタンス化
train_dataset = TitleDocDataset(train_df, tokenizer, texts, max_length=512)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# モデルのインスタンス化とデバイスへの移動
model = TitleDocClassifier(bert_model).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
# モデルの状態をロード
#model.load_state_dict(torch.load('model_epoch_japanese_0_35000.pt', map_location=device))

# 訓練ループ
for epoch in range(3):
    model.train()
    for i, data in enumerate(train_loader):
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        targets = data['targets'].to(device)

        # スコアの計算
        scores = model(ids, mask).squeeze()

        # 損失の計算
        loss = nn.BCEWithLogitsLoss()(scores, targets.float())

        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 定期的なモデルの保存
        if i % 500 == 0:
            print(f'Epoch: {epoch}, Batch: {i+35000}, Loss: {loss.item()}')
            torch.save(model.state_dict(), f'model_epoch_japanese_{epoch}_{i+35000}.pt')
