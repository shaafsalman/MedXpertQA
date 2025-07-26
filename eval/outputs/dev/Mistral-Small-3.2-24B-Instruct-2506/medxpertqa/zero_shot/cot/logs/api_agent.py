from openai import OpenAI
from typing import List
from model.base_agent import LLMAgent
import base64
import traceback
import time
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import io
from PIL import Image as PILImage
import os

def encode_image(image_path):
    if image_path:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    else:
        return "No image inputs"

def create_robust_session():
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def fetch_image_with_retry(image_url, max_attempts=3, timeout=30):
    session = create_robust_session()
    
    for attempt in range(max_attempts):
        try:
            response = session.get(image_url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"Failed to download image after {max_attempts} attempts: {e}")
                return None
            print(f"Attempt {attempt+1} failed. Retrying...")
            time.sleep(2 ** attempt)
    
    return None

class APIAgent(LLMAgent):
    def __init__(self, model_name, temperature=0) -> None:
        super().__init__(model_name, temperature)

        if "o3" in model_name or "o1" in model_name or "qwq" in model_name.lower() or "qvq" in model_name.lower() or "reasoner" in model_name.lower():
            self.max_tokens = 8192
        else:
            self.max_tokens = 2048

        if model_name in [
            "o3-mini-2025-01-31",
            "o1-2024-12-17",
            "o1-preview-2024-09-12",
            "o1-mini-2024-09-12",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini",

            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",

            "gemini-1.5-pro",
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash",
            "gemini-2.0-pro-exp-02-05",
        ]:
            print("OpenAI")
            self.client = OpenAI(
                api_key="[API_KEY]",
                base_url="[BASE_URL]",
            )
        elif model_name in [
            "deepseek-chat",
            "deepseek-reasoner",
        ]:
            print("DeepSeek")
            self.client = OpenAI(
                api_key="[API_KEY]",
                base_url="https://api.deepseek.com/v1",
            )
        elif model_name in [
            "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            "Mistral-Small-3.2-24B-Instruct-2506",
            "Mistral-24B",
            "Zaynoid/Mistral-24B"
        ]:
            print("Mistral vLLM")
            from vllm import LLM, SamplingParams
            import os
            
            os.environ['VLLM_USE_V1'] = '0'
            
            self.client = LLM(
                model=model_name,
                tensor_parallel_size=4,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
                tokenizer_mode="mistral",      
                config_format="mistral",       
                load_format="mistral", 
            )
            self.sampling_params = SamplingParams(
                max_tokens=self.max_tokens, 
                temperature=self.temperature if self.temperature > 0 else 0.15
            )
            self.is_vllm = True
        else:
            raise ValueError("Model not supported")

    def get_response(self, messages: List[dict]) -> str:
        if hasattr(self, 'is_vllm') and self.is_vllm:
            try:
                outputs = self.client.chat(messages, sampling_params=self.sampling_params)
                response = outputs[0].outputs[0].text
                response = "\n".join(line for line in response.splitlines() if line.strip())
                log_probs = []
                return response, log_probs
            except Exception as e:
                print(f"vLLM error: {e}")
                print(traceback.format_exc())
                return "No answer provided.", []
        
        elif ("o3" in self.model_name) or ("o1" in self.model_name) or ("deepseek-reasoner" in self.model_name):
            messages = [m for m in messages if m["role"] != "system"]
            for _ in range(10):
                try:
                    completion = self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_name,
                        max_completion_tokens=self.max_tokens,
                        seed=0
                    )
                    response = completion.choices[0].message.content
                    break
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    time.sleep(1)
                    response = "No answer provided."
        else:
            for _ in range(10):
                try:
                    completion = self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        logprobs=True,
                        seed=0,
                    )
                    response = completion.choices[0].message.content
                    break
                except Exception as e:
                    if "bad_response_status_code" in str(e):
                        print("Bad Response")
                        response = "No answer provided: bad_response."
                        break
                    elif "content_filter" in str(e):
                        print("Content Filter")
                        response = "No answer provided: content_filter."
                        break
                    else:
                        print(e)
                        print(traceback.format_exc())
                        time.sleep(1)
                        response = "No answer provided."
        
        try:
            log_probs = completion.choices[0].logprobs.content
            log_probs = [token_logprob.logprob for token_logprob in log_probs]
        except Exception as e:
            log_probs = []
        return response, log_probs

    def image_content(self, img_path: str) -> dict:
        img_path = img_path.strip()
        if hasattr(self, 'is_vllm') and self.is_vllm:
            if img_path.startswith("http"):
                try:
                    image_data = fetch_image_with_retry(img_path)
                    if image_data is None:
                        return {"type": "text", "text": "[ERROR: Could not download image]"}
                    
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                except Exception as e:
                    return {"type": "text", "text": f"[ERROR: Failed to process image: {str(e)}]"}
            else:
                return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}}
        else:
            if img_path.startswith("http"):
                return {"type": "image_url", "image_url": {"url": img_path}}
            else:
                return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}}

    def generateImageResponse(self, prompt, image_url):
        if hasattr(self, 'is_vllm') and self.is_vllm:
            SYSTEM_PROMPT = "You are a helpful medical assistant. Provide accurate, evidence-based information in response to the following question. Organize the response with clear hierarchical headings and include a conclusion if necessary."
            
            try:
                image_data = fetch_image_with_retry(image_url)
                if image_data is None:
                    return "ERROR: Could not download the image. The service might be unavailable or the image URL may be invalid."
                
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                data_url = f"data:image/jpeg;base64,{image_base64}"
                
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ]
                
                response, _ = self.get_response(messages)
                return response
                
            except Exception as e:
                error_details = traceback.format_exc()
                return f"ERROR: An exception occurred while generating the response: {str(e)}\n\nDetails: {error_details}"
        else:
            image_content = self.image_content(image_url)
            messages = [
                {"role": "system", "content": "You are a helpful medical assistant."},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        image_content
                    ]
                }
            ]
            response, _ = self.get_response(messages)
            return response