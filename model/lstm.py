import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim

# class FlightPhaseLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
#         super(FlightPhaseLSTM, self).__init__()
        
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # LSTM Layer
#         # batch_first=True: 입력이 (Batch, Seq, Feature) 순서
#         self.lstm = nn.LSTM(
#             input_size=input_size, 
#             hidden_size=hidden_size, 
#             num_layers=num_layers, 
#             batch_first=True, 
#             dropout=dropout if num_layers > 1 else 0
#         )
        
#         # Classifier (Fully Connected)
#         self.fc = nn.Linear(hidden_size, num_classes)
        
#     def forward(self, x):
#         # x shape: (Batch, Window_Size, Input_Size)
        
#         # 초기 Hidden State와 Cell State (0으로 초기화)
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
#         # LSTM 순전파
#         # out shape: (Batch, Window_Size, Hidden_Size)
#         out, _ = self.lstm(x, (h0, c0))
        
#         # Many-to-One: 마지막 Time Step의 결과만 사용
#         last_out = out[:, -1, :] 
        
#         # 분류
#         logits = self.fc(last_out)
#         return logits


class FlightPhaseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(FlightPhaseLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classifier
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, hidden=None):
        # hidden이 None이면 PyTorch LSTM이 알아서 0으로 초기화해줍니다.
        # hidden이 들어오면 그 기억을 이어서 사용합니다.
        
        # out: (Batch, Seq, Hidden)
        # next_hidden: (h_n, c_n) 튜플 -> 다음 스텝으로 넘겨줄 기억
        out, next_hidden = self.lstm(x, hidden)
        
        # Many-to-One: 마지막 타임스텝 사용
        last_out = out[:, -1, :] 
        
        logits = self.fc(last_out)
        
        # 결과값과 함께 갱신된 hidden state도 반환
        return logits, next_hidden