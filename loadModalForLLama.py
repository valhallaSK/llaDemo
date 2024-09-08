import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"}
]
# 检测是否成功调用GPU
if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available.")


while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting the chat.")
        break

    messages.append({"role": "user", "content": user_input})
    input_text = "".join([f'{message["role"]}: {message["content"]}\n' for message in messages])

    outputs = pipeline(
        input_text,
        max_new_tokens=256,
        do_sample=True,
        num_return_sequences=1,
    )
    generated_text = outputs[0]["generated_text"]
    print(f"Pirate Bot: {generated_text}")
    messages.append({"role": "assistant", "content": generated_text})
