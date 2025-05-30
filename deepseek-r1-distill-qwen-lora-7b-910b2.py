import mindspore
import mindnlp
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindnlp.engine import TrainingArguments, Trainer
from mindnlp.dataset import load_dataset, BaseMapFunction
import os
import sys
import time
import mindspore as ms
import numpy as np
import random
from mindnlp.engine.utils import PREFIX_CHECKPOINT_DIR
from mindnlp.configs import SAFE_WEIGHTS_NAME
from mindnlp.engine.callbacks import TrainerCallback, TrainerState, TrainerControl
from mindspore._c_expression import disable_multi_thread

# 加载数据集
dataset = load_dataset(path="json", data_files="./huanhuan.json")
# 获取数据列
print(dataset.get_col_names())
# 查看数据
for instruction, input, output in dataset.create_tuple_iterator():
    print("instruction: ", instruction)
    print("input: ", input)
    print("output: ", output)
    break

# 实例化tokenizer
tokenizer = AutoTokenizer.from_pretrained("MindSpore-Lab/DeepSeek-R1-Distill-Qwen-7B", mirror="modelers", use_fast=False, ms_type=mindspore.bfloat16)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# test_tokenized = tokenizer('你好')
# print(test_tokenized.keys())
# 定义数据处理逻辑

def process_func(instruction, input, output):
    MAX_SEQ_LENGTH = 256  # 最长序列长度
    input_ids, attention_mask, labels = [], [], []
    # 首先生成user和assistant的对话模板
    # User: instruction + input
    # Assistant: output
    formatted_instruction = tokenizer(f"User: {instruction}{input}\n\n", add_special_tokens=False)
    formatted_response = tokenizer(f"Assistant: {output}", add_special_tokens=False)
    # 最后添加 eos token，在deepseek-r1-distill-qwen的词表中， eos_token 和 pad_token 对应同一个token
    # User: instruction + input \n\n Assistant: output + eos_token
    input_ids = formatted_instruction["input_ids"] + formatted_response["input_ids"] + [tokenizer.pad_token_id]
    # 注意相应
    attention_mask = formatted_instruction["attention_mask"] + formatted_response["attention_mask"] + [1]
    labels = [-100] * len(formatted_instruction["input_ids"]) + formatted_response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_SEQ_LENGTH:
        input_ids = input_ids[:MAX_SEQ_LENGTH]
        attention_mask = attention_mask[:MAX_SEQ_LENGTH]
        labels = labels[:MAX_SEQ_LENGTH]

    # 填充到最大长度
    padding_length = MAX_SEQ_LENGTH - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
    attention_mask = attention_mask + [0] * padding_length  # 填充的 attention_mask 为 0
    labels = labels + [-100] * padding_length  # 填充的 label 为 -100
    
    return input_ids, attention_mask, labels

# 查看预处理后的数据
formatted_dataset = dataset.map(operations=[process_func], 
                                input_columns=['instruction', 'input', 'output'], 
                                output_columns=["input_ids", "attention_mask", "labels"])


for input_ids, attention_mask, labels in formatted_dataset.create_tuple_iterator():
    print(tokenizer.decode(input_ids))
    break

tokenizer.decode(tokenizer.pad_token_id)

from mindnlp.transformers import GenerationConfig

model = AutoModelForCausalLM.from_pretrained("MindSpore-Lab/DeepSeek-R1-Distill-Qwen-7B", mirror="modelers", ms_dtype=mindspore.bfloat16)
model.generation_config = GenerationConfig.from_pretrained("MindSpore-Lab/DeepSeek-R1-Distill-Qwen-7B")
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model

# model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法


from mindnlp.peft import LoraConfig, TaskType, get_peft_model, PeftModel

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# print(model)

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )       

        # save adapter weights
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path, safe_serialization=True)

        # remove base model safetensors to free more space
        base_model_path = os.path.join(checkpoint_folder, SAFE_WEIGHTS_NAME)
        os.remove(base_model_path) if os.path.exists(base_model_path) else None

        return control



args = TrainingArguments(
    output_dir="./output/DeepSeek_7b_424",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=1000,
    learning_rate=1e-4,
    save_on_each_node=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=formatted_dataset,
    callbacks=[SavePeftModelCallback],
)

trainer.train()

text = "你是谁"
inputs = tokenizer(f"User: {text}\n\n", return_tensors="ms")
outputs = model.generate(**inputs, max_new_tokens=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
