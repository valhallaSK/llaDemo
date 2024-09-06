from model import SimpleLLaMA
import torch

def load_model():
    vocab = {'你': 0, '瞅': 1, '啥': 2, '瞅你': 3, '咋': 4, '地': 5, '<pad>': 6}
    vocab_size = len(vocab)

    d_model = 512
    nhead = 8
    num_layers = 6
    dim_feedforward = 2048

    model = SimpleLLaMA(vocab_size, d_model, nhead, num_layers, dim_feedforward)
    
    model.load_state_dict(torch.load("simple_model_demo.pth"))
    model.eval()

    return model, vocab

def infer(input_sentence):
    model, vocab = load_model()
    input_idx = torch.tensor([[vocab[word] for word in input_sentence]], dtype=torch.long)

    with torch.no_grad():
        output = model(input_idx)
        predicted_tokens = torch.argmax(output, dim=-1).squeeze().tolist()

    predicted_words = [list(vocab.keys())[list(vocab.values()).index(idx)] for idx in predicted_tokens]
    return ''.join(predicted_words)

if __name__ == "__main__":

    while True:
        user_input = input("请输入你的问题 : ")
        if user_input == 'exit':
            print("退出程序。")
            break
        input_words = list(user_input) 
        answer = infer(input_words)
        print(f"模型回答: {answer}")