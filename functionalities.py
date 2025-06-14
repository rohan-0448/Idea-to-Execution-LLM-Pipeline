import pypdf
import re
import yaml
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def get_response(self, prompt: str) -> str:
        pass

class GeminiLLM(BaseLLM):
    def __init__(self, model):
        self.model = model

    def get_response(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text
    
class GPTLLM(BaseLLM):
    def __init__(self, model):
        self.model = model

    def get_response(self, prompt: str) -> str:
        response = self.model.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2500,
            temperature=0.7,
        )
        return response.choices[0].message.content
    
class ClaudeLLM(BaseLLM):
    def __init__(self, model):
        self.model = model

    def get_response(self, prompt: str) -> str:
        response = self.model.messages.create(
            model=self.model,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
def check_llm(llm_name):
    return llm_name.split("-")[0].lower()
    
def get_llm_response(model, prompt):
    """Function to get the response from the LLM model"""
    
    response = model.get_response(prompt)
    return response

def read_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_idea(path):

    """Function to get high level idea generated from the plan-generation pipeline"""

    with open(path, 'r') as file:
        text = file.read()

    return text

def get_code(path):

    """Function to import baseline code as text"""

    with open(path, 'r') as file:
        text = file.read()

    return text

def read_pdf(pdf_path):

    """Extract text from the pdf file"""

    reader = pypdf.PdfReader(pdf_path) # create a pdf reader object
    num_pages = len(reader.pages)

    text = ""
    for i in range(num_pages):
        text += reader.pages[i].extract_text()

    return text

def summarize_paper(model, text, template_path):

    """Summarize the paper using the AI model"""

    with open(template_path, 'r') as file:
        prompt_template = file.read()

    prompt = prompt_template.format(text=text)
    response = get_llm_response(model, prompt)

    return response

def analyse_summary(model, text, context, template_path):
    
    """Analyse the summary and extract key points"""

    text = f"{text} + '\nContext: {context}" # append summary to the end of the research paper

    with open(template_path, 'r') as file:
        prompt_template = file.read() 

    prompt = prompt_template.format(text=text)
    response = get_llm_response(model, prompt)

    return response

def model_requirement(model, text, prompt_template):
    """Function to get required model name"""

    with open(prompt_template, 'r') as file:
        prompt_template = file.read()

    prompt = prompt_template.format(text=text)
    response = get_llm_response(model, prompt)

    return response

def model_type(model, text, prompt_template):
    """Function to get model type"""

    with open(prompt_template, 'r') as file:
        prompt_template = file.read()

    prompt = prompt_template.format(text=text)
    response = get_llm_response(model, prompt)

    return response

def generate_methodology(model, text, context, idea, prompt_template):

    """Generate methodology for the research paper"""

    text = f"{text}\nContext: {context}\nIdea: {idea}"
    
    with open(prompt_template, 'r') as file:
        prompt_template = file.read()

    prompt = prompt_template.format(text=text)
    response = get_llm_response(model, prompt)

    return response

def generate_high_level_plan(model, text, context, idea, prompt_template):

    """Function to create high level plan"""

    # text = f"{text} + '\nContext: {context} + '\nIdea: {idea} + '\nTrain Baseline Code: {train_baseline_code} + '\nTest Baseline Code: {test_baseline_code}" 
    text = f"{text}\nContext: {context}\nIdea: {idea}"

    with open(prompt_template, 'r') as file:
        prompt_template = file.read()

    prompt = prompt_template.format(text=text)
    response = get_llm_response(model, prompt)

    return response

def get_text(path):

    """Function to get high level idea generated from the plan-generation pipeline"""

    with open(path, 'r') as file:
        text = file.read()

    return text

def clean_code_block(code: str) -> str:
    """Remove markdown code block syntax using regex"""
    # Match both ```python and ``` markers with any whitespace
    pattern = r'^\s*```python\s*|\s*```\s*$'
    return re.sub(pattern, '', code, flags=re.MULTILINE).strip()

def baseline_test_code(model, model_given, high_level_plan, base_test_code, prompt_template):
    """Function to generate a new testing code based on the given model and research paper analysis."""
    
    with open(prompt_template, 'r') as file:
        prompt_template = file.read()

    prompt = prompt_template.format(model_given=model_given, high_level_plan=high_level_plan, base_code=base_test_code)
    response = get_llm_response(model, prompt)

    return response

def baseline_train_code(model, model_given, high_level_plan, base_test_code, prompt_template):
    """Function to generate a new training code based on the given model and research paper analysis."""

    with open(prompt_template, 'r') as file:
        prompt_template = file.read()

    prompt = prompt_template.format(model_given=model_given, high_level_plan=high_level_plan, base_code=base_test_code)
    response = get_llm_response(model, prompt)

    return response

def ask_reviewer_train(model, high_level_plan, train_code, problem_encountered, prompt_template):
    # make new reviewer and ask for wht could be wrong in this code

    """Function to ask the reviewer for feedback on the generated code and the error encountered"""

    with open(prompt_template, 'r') as file:
        prompt_template = file.read()

    prompt = prompt_template.format(high_level_plan=high_level_plan, train_code=train_code, problem_encountered=problem_encountered)
    response = get_llm_response(model, prompt)

    return response

def ask_reviewer_test(model, high_level_plan, train_code, test_code, problem_encountered, prompt_template):

    """Function to ask the reviewer for feedback on the generated code and the error encountered"""

    with open(prompt_template, 'r') as file:
        prompt_template = file.read()

    prompt = prompt_template.format(high_level_plan=high_level_plan, train_code=train_code, test_code=test_code, problem_encountered=problem_encountered)
    response = get_llm_response(model, prompt)

    return response


def remake_training_code(model, idea, error, old_code, prompt_template):
    """Function for error handling in training code"""

    with open(prompt_template, 'r') as file:
        prompt_template = file.read()

    prompt = prompt_template.format(idea=idea, error=error, old_code=old_code)
    response = get_llm_response(model, prompt)

    return response

def remake_testing_code(model, idea, error, train_code, old_testing_code, prompt_template):
    """Function for error handling in testing code"""

    with open(prompt_template, 'r') as file:
        prompt_template = file.read()

    prompt = prompt_template.format(idea=idea, error=error, train_code=train_code, old_code=old_testing_code)
    response = get_llm_response(model, prompt)

    return response

def get_additional_libraries(model, file, template_path):
    """Function to split the generated code into train.py and test.py"""

    with open(file, 'r') as f:
        text = f.read()

    with open(template_path, 'r') as file:
        prompt_template = file.read() 

    prompt = prompt_template.format(text=text)
    response = get_llm_response(model, prompt)

    return response

def half_report(model, high_level_plan, issue_encountered, prompt_template):
    """Function to genrate half-report when final code wasn't running with steps to follow to work out the code"""

    with open(prompt_template, 'r') as file:
        prompt_template = file.read()

    prompt = prompt_template.format(high_level_plan=high_level_plan, issue_encountered=issue_encountered)
    response = get_llm_response(model, prompt)

    return response
    
def full_report(model, high_level_plan, training_code, log_report, prompt_template):
    """Function to genrate full-report with final code and conclusion"""

    with open(prompt_template, 'r') as file:
        prompt_template = file.read()

    prompt = prompt_template.format(high_level_plan=high_level_plan, training_code=training_code, log_report=log_report)
    response = get_llm_response(model, prompt)

    return response

def analyse_log(model, log_report, prompt_template):
    """Function to check the log output when running final Training Code"""

    with open(prompt_template, 'r') as file:
        prompt_template = file.read()

    prompt = prompt_template.format(log_report=log_report)
    response = get_llm_response(model, prompt)

    return response

def sequential_check(text):
    text = text.lower()

    sequential_count = text.count("sequential")
    flat_count = text.count("flat")

    return sequential_count > flat_count