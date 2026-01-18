import os
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams



model_path = "/Model/Qwen-72B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = LLM(model=model_path, tensor_parallel_size=8)

sampling_params = SamplingParams(temperature=0.6, max_tokens=3500)


def load_user_input(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def create_message(prompt: str) -> str:
    
    system_content = load_user_input("User_Prompt.txt")
    system_message = {"role": "system", "content": system_content}

    user_command = 'Please generate detailed data in JSON format based on the given reaction.'
    user_input = prompt + '\n' + user_command

    messages = [
        system_message,
        {"role": "user", "content": user_input}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def load_prompts_from_file(file_path, start_line, end_line):
    
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return lines[start_line:end_line]



def save_generated_output(output_text, file_path):
    
    try:

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(output_text, file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error: {id}")



def main():
    
    input_file_path = "train_label_mix.txt"
    write_file_path = "/data/CoT/"

    os.makedirs(write_file_path, exist_ok=True)

    start_line, end_line = 100, 216000
    prompts = load_prompts_from_file(input_file_path, start_line, end_line)

    batch_size = 8
    total_batches = len(prompts) // batch_size

    for i in range(total_batches):
        batch_prompts = prompts[i * batch_size: (i + 1) * batch_size]
        batch_texts = [create_message(prompt.strip()) for prompt in batch_prompts]

        outputs = llm.generate(batch_texts, sampling_params)

        for j, output in enumerate(outputs):
            file_id = start_line + i * 8 + j + 1
            generated_text = output.outputs[0].text

            file_path = os.path.join(write_file_path, f"{file_id}.txt")

            save_generated_output(generated_text, file_path)



if __name__ == "__main__":
    
    main()
