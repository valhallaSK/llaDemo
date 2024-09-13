from datasets import load_dataset
from transformers import LlamaForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model



# 加载自定义的问答数据集
dataset = load_dataset('json', data_files='trainData.json')
train_test_split = dataset['train'].train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# 加载预训练的 LLaMA 3.1 模型和分词器
model_name = "/openbayes/input/input0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")

# 设置 pad_token
tokenizer.pad_token = tokenizer.eos_token

# 数据预处理函数
def preprocess_function(examples):
    # 标记化输入 (问题)
    inputs = tokenizer(examples['question'], max_length=128, truncation=True, padding="max_length")
    # 标记化输出 (答案)
    targets = tokenizer(examples['answer'], max_length=128, truncation=True, padding="max_length")
    inputs['labels'] = targets['input_ids']
    return inputs


# 对训练数据进行预处理
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_eval = eval_dataset.map(preprocess_function, batched=True)
# 配置 LoRA 参数
lora_config = LoraConfig(
    r=16,  
    lora_alpha=32,  
    target_modules=["q_proj", "v_proj", "k_proj"], 
    lora_dropout=0.15, 
    task_type="CAUSAL_LM"  
)

# 使用 LoRA 配置微调 LLaMA 3.1
model = get_peft_model(model, lora_config)
# 设置训练参数
training_args = TrainingArguments(
    output_dir="./llama3_finetuned",
    per_device_train_batch_size=4,  # 增加批量大小以加快训练速度
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    logging_steps=10,  
    num_train_epochs=30,  
    save_steps=50,
    learning_rate=5e-4, 
    weight_decay=0.01,  # 权重衰减，用于防止过拟合
    save_total_limit=2, 
    fp16=True, 
    push_to_hub=False  # 不推送到 Hugging Face Hub
)

# 创建Trainer对象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,  # 提供训练集
    eval_dataset=tokenized_eval,    # 提供验证集
    tokenizer=tokenizer
)

# 开始微调
trainer.train()

# 保存微调后的模型
trainer.save_model("./llama3_finetuned")
print("模型已保存")