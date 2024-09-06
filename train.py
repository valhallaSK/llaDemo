from model import SimpleLLaMA
import torch
import torch.nn as nn
import torch.optim as optim

vocab = {'你': 0, '瞅': 1, '啥': 2, '瞅你': 3, '咋': 4, '地': 5, '<pad>': 6}
vocab_size = len(vocab)

question = ['你', '瞅', '啥']
answer = ['瞅你', '咋', '地']

question_idx = torch.tensor([[vocab[word] for word in question]], dtype=torch.long)  # [1, 3]
answer_idx = torch.tensor([[vocab[word] for word in answer]], dtype=torch.long)  # [1, 3]

max_len = max(len(question_idx[0]), len(answer_idx[0]))
padded_question = torch.cat([question_idx, torch.tensor([[vocab['<pad>']] * (max_len - len(question_idx[0]))], dtype=torch.long)], dim=1)
padded_answer = torch.cat([answer_idx, torch.tensor([[vocab['<pad>']] * (max_len - len(answer_idx[0]))], dtype=torch.long)], dim=1)

d_model = 512
nhead = 8
num_layers = 6
dim_feedforward = 2048

model = SimpleLLaMA(vocab_size, d_model, nhead, num_layers, dim_feedforward)

criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    output = model(padded_question).view(-1, vocab_size)
    loss = criterion(output, padded_answer.view(-1))
    
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

torch.save(model.state_dict(), "simple_model_demo.pth")
print("训练完成并保存模型")