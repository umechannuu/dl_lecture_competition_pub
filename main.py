import random
import time
from statistics import mode

import pandas
import torchvision

import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertTokenizer, BertModel
from torch.optim import lr_scheduler
#from scipy.stats import mode
import time



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer
        #self.tokenizer = tokenizer

        #answerの辞書を作成
        self.answer2idx = {}
        self.idx2answer = {}
        
        
        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
       
        self.answer2idx = dataset.answer2idx
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        input_question = self.df["question"][idx]
        
        
        
        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）
        
        

            return image, input_question, torch.Tensor(answers), int(mode_answer_idx)

        else:
            return image, input_question

    def __len__(self):
        return len(self.df)

def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)

class VQAModel(nn.Module):
    def __init__(self, n_answer: int, tokenizer):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.bert_model = BertModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float32, attn_implementation="sdpa")
        for param in self.bert_model.parameters():
            param.requires_grad = False
            
        for param in self.bert_model.encoder.layer[-2:].parameters():
            param.requires_grad = True
            
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)


        self.fc = nn.Sequential(
            nn.Linear(512+768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question):
        N = image.shape[0]
        image_feature = self.resnet(image)  
        
        
        with torch.no_grad():
            question = self.tokenizer(
                question,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(image.device)
            question_feature = self.bert_model(**question).last_hidden_state[:, 0, :]
        
        
        
        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x
    
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question, answers.to(device), mode_answer.to(device)

        pred = model(image, question)
    
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question, answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform_1 = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # ランダムリサイズクロップ
                transforms.RandomHorizontalFlip(),  # ランダム水平反転
                transforms.RandomRotation(10),  
                transforms.ToTensor()])
    
    transform_2 = transforms.Compose([
                transforms.Resize((224, 224)),  # リサイズ
                transforms.ToTensor()])
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = VQADataset(df_path="/kaggle/input/data-vqa2/train.json", image_dir="/kaggle/input/data-vqa2/train/train", transform=transform_1)
    test_dataset = VQADataset(df_path="/kaggle/input/data-vqa2/valid.json", image_dir="/kaggle/input/data-vqa2/valid/valid", transform=transform_2, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = VQAModel(n_answer=len(train_dataset.answer2idx),tokenizer=tokenizer).to(device)
    

    
    criterion = nn.CrossEntropyLoss()

    num_epoch = 15
    
    bert_params = list(model.bert_model.encoder.layer[-2:].parameters())
    resnet_params = list(model.resnet.layer4.parameters())
    fc_params = list(model.fc.parameters())

    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': 5e-5},
        {'params': resnet_params, 'lr': 1e-5},
        {'params': fc_params, 'lr': 1e-4}
    ])
    
    criterion = nn.CrossEntropyLoss()

    
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")
    
    torch.save(optimizer.state_dict(), "optim.pth")

    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)

if __name__ == "__main__":
    main()


