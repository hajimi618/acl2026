import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import time


client = OpenAI(
    api_key="XXX",
    base_url="XXX",
)


A_file = "/Retro_Predict/reaction_input.txt"
B_file = "/Retro_Predict/reaction_product.txt"
C_file = "/Retro_Predict/ORD_train_Mix.txt"


BATCH_SIZE = 8
MAX_RETRY = 3


with open(A_file, "r", encoding="utf-8") as fa, \
     open(B_file, "r", encoding="utf-8") as fb, \
     open(C_file, "r", encoding="utf-8") as fc:

    input_lines = [x.strip() for x in fa]
    product_lines = [x.strip() for x in fb]
    Mix_lines = [x.strip() for x in fc]

assert len(input_lines) == len(product_lines) == len(Mix_lines), "Error"
N = len(input_lines)


def build_prompt(input_line, product_line, Mix_line):

    prompt = f"""
You are an expert in organic chemistry.

Given the complete standard reaction equation:
{Mix_line}

The reaction inputs are:
{input_line}

The products are:
{product_line}

You need to reason through the process. Given the target product, pretend not to know the reaction inputs, and deduce the required reaction inputs (which may include reactants, reagents, solvents, etc.).
Do not reveal the standard answer prematurely. The final result must match the standard answer.

The reasoning content must strictly follow the format below, with each reasoning step title placed in <>.

<think>
1. <Analyze the target product>
2. <Analyze bond-disconnection strategy>
3. <Predict reaction inputs>
4. <Validate the synthetic route>
5. <Confirm reaction inputs> (reaction inputs given in the form <chem>SMILES</chem>)
</think>
"""
    return prompt


def process_one(idx, input_line, product_line, Mix_line):
    
    output_dir = '/Retro_Predict/Reason/'
    
    prompt = build_prompt(input_line, product_line, Mix_line)

    for attempt in range(1, MAX_RETRY + 1):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                        ],
                model="gpt-4o-mini",
)

            output = chat_completion.choices[0].message.content
            out_path = os.path.join(output_dir, f"{idx}.txt")

            with open(out_path, "w", encoding="utf-8") as fout:
                fout.write(output)

            return f"OK: line {idx}"

        except Exception as e:
            print(f"Error at line {idx}, attempt {attempt}: {e}")
            time.sleep(1)

    return f"FAILED: line {idx}"


with ThreadPoolExecutor(max_workers=BATCH_SIZE) as exe:
    futures = [
        exe.submit(process_one, idx, input_line, product_line, Mix_line)
        for idx, (input_line, product_line, Mix_line) in enumerate(zip(input_lines, product_lines, Mix_lines), start=1)
    ]

    # 打印进度
    for f in as_completed(futures):
        print(f.result())

