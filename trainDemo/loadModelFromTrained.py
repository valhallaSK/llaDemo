from transformers import pipeline,AutoModelForCausalLM,AutoTokenizer

# 加载微调后的模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained("./llama3_finetuned", load_in_4bit=True, device_map="auto")
# model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained('/openbayes/input/input0')
tokenizer.pad_token_id = tokenizer.eos_token_id
# 创建文本生成 pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 测试生成
prompt = "What does Liyinkai do?"
result = generator(prompt, max_length=200)

# 输出生成的结果
print(result[0]['generated_text'])