class TestDataset(Dataset):
    def __init__(self, data, tokenizer, texts, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.texts = texts
        self.max_length = max_length
        self.pairs = self.create_pairs(data)

    def create_pairs(self, data):
        pairs = []
        for item in data:
            title_id = item['title_id']
            title_text = self.texts[title_id]
            for doc_id in item['candidates']:
                doc_text = self.texts[doc_id]
                pairs.append((title_id, doc_id, title_text, doc_text))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        title_id, doc_id, title_text, doc_text = self.pairs[idx]
        inputs = self.tokenizer.encode_plus(
            title_text,
            doc_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'title_id': title_id,
            'doc_id': doc_id,
            'ids': inputs['input_ids'].flatten(),
            'mask': inputs['attention_mask'].flatten()
        }


test_data = load_data('test_q.json')
# モデルのインスタンス化とデバイスへの移動
model = TitleDocClassifier(bert_model).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)

# モデルの状態をロード
#model.load_state_dict(torch.load('model_epoch_japanese_0_31000.pt', map_location=device))

# JSONファイルから読み込んだデータをDataFrameに変換
test_df = pd.DataFrame(test_data)


# JSONファイルから読み込んだデータをDataFrameに変換
test_df = pd.DataFrame(test_data)

# TestDatasetクラスのインスタンス化
test_dataset = TestDataset(test_data, tokenizer, texts, max_length=512)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ソフトマックス関数の定義
softmax = nn.Softmax(dim=1)

model.eval()  # モデルを評価モードに設定
title_doc_scores = {}  # タイトルIDと対応するドキュメントIDおよびスコアを格納する辞書

with torch.no_grad():
    for data in test_loader:
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        title_ids = data['title_id']
        doc_ids = data['doc_id']
        outputs = model(ids, mask).squeeze()  # 出力を1次元に変更
        relevant_scores = outputs.cpu().numpy()

        for title_id, doc_id, score in zip(title_ids, doc_ids, relevant_scores):
            if title_id not in title_doc_scores:
                title_doc_scores[title_id] = []
            title_doc_scores[title_id].append((doc_id, score))

# タイトルIDごとにドキュメントをスコア順に並べ替え
sorted_suggestions = []
for title_id, doc_scores in title_doc_scores.items():
    sorted_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    sorted_doc_ids = [doc_id for doc_id, _ in sorted_docs]
    sorted_suggestions.append({'title_id': title_id, 'candidates': sorted_doc_ids})

# JSONファイルとして保存
with open('suggestion.json', 'w', encoding='utf-8') as file:
    json.dump(sorted_suggestions, file, ensure_ascii=False)
