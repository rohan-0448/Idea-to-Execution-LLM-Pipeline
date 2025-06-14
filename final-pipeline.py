# MAKE IMPORTS

import google.generativeai as genai
import openai
from anthropic import Anthropic
from functionalities import *
import os
import subprocess
import sys

# API KEY

config = read_config('config.yaml')

llm_name = check_llm(config['llm_model']) # get name of the LLM
API_KEY = config['api_key'] # get API key of the LLM

if check_llm(llm_name) == "gemini":
    genai.configure(api_key=API_KEY)
    model = GeminiLLM(genai.GenerativeModel(config['llm_model'])) 
elif check_llm(llm_name) == "gpt":
    client = openai.OpenAI(api_key=API_KEY)
    model = GPTLLM(client)
elif check_llm(llm_name) == "claude":
    claude_model = Anthropic(api_key=API_KEY)
    model = ClaudeLLM(claude_model)
else:
    print(f"Model not suppoted: {check_llm(llm_name)}")
    sys.exit()

# ATTEMPT NAME
attempt_name = config['attempt_name']

print(attempt_name)

# REQUIRED DIRECTORIES

required_dirs = [
    f'logs/{attempt_name}/STEP1',
    f'logs/{attempt_name}/STEP2',
    f'logs/{attempt_name}/STEP3',
    f'logs/{attempt_name}/STEP4',
    f'logs/{attempt_name}/STEP5',
    f'logs/{attempt_name}/STEP6',
    f'logs/{attempt_name}/STEP7',
    f'logs/{attempt_name}/STEP8',
    f'logs/{attempt_name}/STEP9',
    f'logs/{attempt_name}/STEP10',
    f'final_code_files/{attempt_name}',
    f'report/{attempt_name}'
]

for dir_path in required_dirs:
    os.makedirs(dir_path, exist_ok=True)

# SAVE IT INPUT OR PAPER TEXT

paper_names = config['paper_paths']
user_idea_path = config['user_idea_path']

paper_text = ""
for file_path in paper_names:
    if os.path.isfile(file_path):
        # check if the file is a PDF or text file and read accordingly
        if file_path.endswith('.pdf'):
            temp_text = read_pdf(file_path) # read pdf used to read pdf file
        elif file_path.endswith('.txt'):
            temp_text = get_text(file_path) # get text used to read text file
        else:
            print(f"Unsupported file format: {file_path}")
            continue

        paper_text += temp_text + "\n\n"

user_idea = get_idea(user_idea_path)

# DATASET CONTEXT AND NUMIN API INFORMATION

info = """1. Dataset Context: Two types of data are used, training and testing. Data is downloaded using API calls to the Numin Platform. Additional training data is also available on the Numin platform and can be downloaded as described in the train_baseline code given below.
       This data is then saved into the training_data folder. For convenience, this data is already saved in the ./training_data and has naming convention as follows 'df_val_01-Apr-2024'. The data is in csv format. The data is then imported using the pandas library and stored in a dataframe.
       The id column tells us about the stock id and is in the text format. The other features are float values. 
       The last column in case of training data is the target variable. It is converted from the [-1,1] range to [0,4] using the formula y = [int(2 * (score_10 + 1)) for score_10 in y]. Do not assume any additional information about the features of the dataset.
       """

# COMBINING ALL INPUTS

combined_input = paper_text + info + user_idea


# STEP 1 - SUMMARIZE THE INPUT

summary = summarize_paper(model, combined_input, 'prompts_final/summarize-paper-template.txt')
print("STEP 1 - THE INPUT HAS BEEN SUMMARISED...\n\n")

with open(f'logs/{attempt_name}/STEP1/summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)


# STEP 2 - ANALYSE THE INPUT

analysis = analyse_summary(model, combined_input, summary, 'prompts_final/analyze-summary-template.txt') # get analysis
print("STEP 2 - THE INPUT HAS BEEN ANALYSED...\n\n")

with open(f'logs/{attempt_name}/STEP2/analysis.txt', 'w', encoding='utf-8') as f:
    f.write(analysis)


# STEP 3 - GET MODEL REQUIRED

model_required = model_requirement(model, combined_input, 'prompts_final/model-requirement.txt')

print("STEP 3 - MODEL HAS BEEN FINALISED...\n\n")

with open(f'logs/{attempt_name}/STEP3/model_required.txt', 'w', encoding='utf-8') as f:
    f.write(model_required)


# STEP 4 - GET MODEL TYPE

model_type_required = model_type(model, model_required, 'prompts_final/model-type.txt')

print("STEP 4 - MODEL TYPE (FLAT or SEQUENTIAL) HAS BEEN FINALISED...\n\n")

with open(f'logs/{attempt_name}/STEP4/model_type.txt', 'w', encoding='utf-8') as f:
    f.write(model_type_required)


# STEP 5 - GENERATE METHODOLOGY

combined_input = summary + '\n\n' + analysis

methodology = generate_methodology(model, combined_input, combined_input, info, 'prompts_final/generate-methodology-template.txt') # get methodology
print("STEP 5 - THE METHODOLOGY HAS BEEN GENERATED...\n\n")

with open(f'logs/{attempt_name}/STEP5/methodology.txt', 'w', encoding='utf-8') as f:
    f.write(methodology)


# STEP 6 - GENERATE HIGH LEVEL PLAN

combined_input = summary + '\n\n' + methodology

high_level_plan = generate_high_level_plan(model, combined_input, combined_input, info, 'prompts_final/high-level-plan-template.txt')
print("STEP 6 - THE HIGH LEVEL PLAN HAS BEEN GENERATED...\n\n")

with open(f'logs/{attempt_name}/STEP6/high_level_plan.txt', 'w', encoding='utf-8') as f:
    f.write(high_level_plan)


# STEP 7 - BASE CODES FINALISATION

if(sequential_check(model_type_required)):
    base_train_code_directory = 'base_codes/train-lstm.py'
    base_test_code_directory = 'base_codes/test-lstm.py'
else:
    base_train_code_directory = 'base_codes/train-mlp.py'
    base_test_code_directory = 'base_codes/test-mlp.py'

with open(base_train_code_directory, "r", encoding="utf-8") as file:
    base_train_code = file.read()

with open(base_test_code_directory, "r", encoding="utf-8") as file:
    base_test_code = file.read()


# STEP 8 - TRAINING CODE GENERATION

train_code = baseline_train_code(model, model_required, high_level_plan, base_train_code, 'prompts_final/generate-train.txt')

with open(f'logs/{attempt_name}/STEP8/training_code.txt', 'w', encoding='utf-8') as f:
    f.write(train_code)

with open(f'final_code_files/{attempt_name}/base_train.py', 'w', encoding='utf-8') as file:
    file.write(clean_code_block(train_code))

with open(f'final_code_files/{attempt_name}/temp_train.py', 'w', encoding='utf-8') as file:
    file.write(clean_code_block(train_code))

print("STEP 8 - TRAINING CODE HAS BEEN GENERATED...\n\n")


# STEP 9 - CODE RUNNING AND ERROR HANDLING LOOP

script_to_run = f"final_code_files/{attempt_name}/temp_train.py"

max_attempts = 15
attempt = 0
flag = 0
infinite_loop = 0

def run_with_timeout(command, timeout_seconds=120):
    """
    Run a command with a timeout and return its output.
    Returns (returncode, stdout, stderr) or raises TimeoutError if timeout occurs.
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Code execution timed out - possible infinite loop or stuck process"

while attempt < max_attempts:
    attempt += 1
    print(f"STEP 9 - TRAINING CODE RUNNING ATTEMPT - {attempt}...\n\n")

    try:
        # RUNNING THE SCRIPT WITH TIMEOUT
        returncode, stdout, stderr = run_with_timeout(["python", script_to_run])

        # STORING ANY ERRORS ENCOUNTERED
        if returncode != 0:
            with open(f'logs/{attempt_name}/STEP9/training_error_{attempt}.txt', "w", encoding="utf-8") as f:
                f.write(f"\n\nAttempt {attempt} Error Log:\n")
                f.write(stderr)
                if returncode == -1:
                    f.write("\n\nCode execution timed out - possible infinite loop or stuck process")
                    infinite_loop = 1
                    break

            print(f"STEP 9 - ERROR ENCOUNTERED! STORED IN error_log_file...\n\n")
        else:
            # NO ERROR EXIT THE LOOP
            print("STEP 9 - TRAINING CODE RAN SUCESSFULLY...\n\n")
            with open(f'final_code_files/{attempt_name}/final_training_code.py', 'w', encoding='utf-8') as file:
                file.write(clean_code_block(new_train))
            flag = 1
            break 

    except Exception as e:
        with open(f'logs/{attempt_name}/STEP9/training_error_{attempt}.txt', "w", encoding="utf-8") as f:
            f.write(f"\n\n Exception:\n{str(e)}\n")

        print(f"STEP 9 - EXCEPTION ENCOUNTERED! STORED IN error_log_file...\n\n")

    # ERROR ENCOUNTERED TAKEN
    with open(f'logs/{attempt_name}/STEP9/training_error_{attempt}.txt', "r", encoding="utf-8") as file:
        problem_encountered = file.read()
    
    # ASK REVIEWER AGENT FOR CHECKING ERROR
    reviewer_output = ask_reviewer_train(model, high_level_plan, train_code, problem_encountered, 'prompts_final/ask-reviewer-for-train-mistake.txt')

    # REMAKE TRAINING CODE USING REVIEWER OUTPUT
    new_train = remake_training_code(model, high_level_plan, reviewer_output, train_code, 'prompts_final/remake-training-code.txt')
    
    # STORE THE NEW TRAINING CODE IN TEMP FILE
    with open(f'final_code_files/{attempt_name}/temp_train.py', 'w', encoding='utf-8') as file:
        file.write(clean_code_block(new_train))

    # LOG TRAINING CODES
    with open(f'logs/{attempt_name}/STEP9/train_attempt_{attempt}.py', 'w', encoding='utf-8') as file:
        file.write(clean_code_block(new_train))

print("STEP 9 - FINISHED ERROR RESOLVING ATTEMPTS...\n\n")


# STEP 10 - GENERATE TESTING CODE USING WORKING TRAINING CODE

t_flag = 0 # FLAG TO CHECK IF TESTING CODE WORKS 
t_infinite_loop = 0
if(flag == 1):
    test_code = baseline_test_code(model, model_required, high_level_plan, base_test_code, 'prompts_final/generate-test.txt')

    with open(f'final_code_files/{attempt_name}/temp_test.py', 'w', encoding='utf-8') as file:
        file.write(clean_code_block(test_code))

    print("STEP 10 - TESTING CODE HAS BEEN GENERATED AND STORED...\n\n")

    # DEBUG TESTING CODE FOR ERRORS
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        print(f"STEP 10 - TESTING CODE RUNNING ATTEMPT - {attempt}...\n\n")

        try:
            returncode, stdout, stderr = run_with_timeout(["python", f"final_code_files/{attempt_name}/temp_test.py"])

            if returncode != 0:
                with open(f'logs/{attempt_name}/STEP10/testing_error_{attempt}.txt', "w", encoding="utf-8") as f:
                    f.write(f"\n\nAttempt {attempt} Error Log:\n")
                    f.write(stderr)
                    if returncode == -1:
                        f.write("\n\nCode execution timed out - possible infinite loop or stuck process")
                        t_infinite_loop = 1
                        break
                print("STEP 10 - ERROR ENCOUNTERED! STORED IN error_log_file...\n\n")
            else:
                print("STEP 10 - TESTING CODE RAN SUCESSFULLY...\n\n")
                t_flag = 1
                with open(f'final_code_files/{attempt_name}/final_testing_code.py', 'w', encoding='utf-8') as file:
                    file.write(clean_code_block(test_code))
                break

        except Exception as e:
            with open(f'logs/{attempt_name}/STEP10/testing_error_{attempt}.txt', "w", encoding="utf-8") as f:
                f.write(f"\n\n Exception:\n{str(e)}\n")
            print(f"STEP 10 - EXCEPTION ENCOUNTERED! STORED IN error_log_file...\n\n")

        problem_encountered = get_text(f'logs/{attempt_name}/STEP10/testing_error_{attempt}.txt')
        reviewer_output = ask_reviewer_test(model, high_level_plan, new_train, test_code, problem_encountered, 'prompts_final/ask-reviewer-for-test-mistake.txt') 

        test_code = remake_testing_code(model, high_level_plan, reviewer_output, new_train, test_code, 'prompts_final/remake-testing-code.txt')

        # store the new testing code in temp file
        with open(f'final_code_files/{attempt_name}/temp_test.py', 'w', encoding='utf-8') as file:
            file.write(clean_code_block(test_code))

        # log testing codes
        with open(f'logs/{attempt_name}/STEP10/test_attempt_{attempt}.py', 'w', encoding='utf-8') as file:
            file.write(clean_code_block(test_code))

    print("STEP 10 - FINISHED ERROR RESOLVING ATTEMPTS...\n\n")

# STEP 11 - RUNNING TRAINING FILE IF NO ERRORS

if flag == 1:
    result = subprocess.run(["python3", f"final_code_files/{attempt_name}/final_training_code.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # CAPTURE THE LOG
    logs = result.stdout + "\n" + result.stderr

    print(logs)

    log_report = analyse_log(model, logs, 'prompts_final/analyse-log.txt')


# STEP 12 - FINAL REPORT + CODES

if(flag == 1 and t_flag == 1):
    # with open(f'final_code_files/{attempt_name}/final_training_code.py', 'r', encoding='utf-8') as f:
    #     final_code = f.read()

    final_code = get_text(f'final_code_files/{attempt_name}/final_training_code.py')
    report = full_report(model, high_level_plan, final_code, log_report,'prompts_final/full-report.txt')
    # final code have been generated the pipeline worked properly
else:
    issue = "Code Generation and Debugging"
    if(infinite_loop==1 or t_infinite_loop == 1):
        issue = "Code is enconuntering an infinte loop "
    report = half_report(model, high_level_plan, issue, 'prompts_final/half-report.txt')
    # issue in the pipeline, make a full report of the findings, a the final code which couldn't work. ask user to figure it out

with open(f'report/{attempt_name}/final_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("REPORT GENERATED AND PIPELINE DISMISSED")   