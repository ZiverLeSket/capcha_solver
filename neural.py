from torch import nn 
import torch
import string
import cv2 as cv

_DEVICE_ = ('cuda')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(38*21, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 35),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(_DEVICE_)
model.load_state_dict(torch.load("model.pth"))

symbols = string.digits + string.ascii_lowercase
symbols = symbols[1:]
sym_map = dict()
for sym_idx, sym in enumerate(symbols):
    data = [0]*35
    data[sym_idx] = 1
    sym_map[sym] = torch.tensor(data, device=_DEVICE_, dtype=torch.float32)

sample = cv.imread("batch1/csoxzm.jpg")
gray_sample = cv.cvtColor(sample, cv.COLOR_BGR2GRAY)
gray_sample = gray_sample/255
cropped_sample = gray_sample[15:53, 9:135]
result = ''
for i in range(6):
    data = torch.from_numpy(cropped_sample[:, i*21:(i+1)*21])
    data = data.reshape([1, 38, 21])
    data = data.to(torch.float32)
    data = data.to(_DEVICE_)
    logits = model(data)
    pred_probab = nn.Softmax(dim=1)(logits)
    pred = pred_probab.argmax(1)
    result += symbols[int(pred)]

print(result) 