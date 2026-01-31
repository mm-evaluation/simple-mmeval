import re
import copy
import torch
import torchvision.transforms as transforms

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"torch_dtype": self.dtype}
        self.default_gen_kwargs = {"max_new_tokens": 512, "do_sample": False}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super().__init__(args)
    
    def load_model(self, args):
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            **self.model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True
        )
        self.model = self.model.to(self.device)
        self.model.tokenizer = self.tokenizer
        self.model.eval()

    def _parse_input(self, msg):
        prompt = msg["prompt"]
        q_chunks = re.split(r'(<image>)', prompt)
        media = copy.deepcopy(msg["media"])

        text_content = ""
        image = None

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                media_file = media.pop(0)
                if isinstance(media_file, Image.Image):
                    image = media_file.convert('RGB')
                else:
                    image = Image.open(media_file).convert('RGB')
                
                transform = transforms.Compose([
                    transforms.Resize((224, 224)), 
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                if isinstance(image, Image.Image):
                    image = transform(image).unsqueeze(0)  
                    image = image.to(self.device)
            else:
                text_content += chunk
        
        return {"text": text_content, "image": image}

    def generate_output(self, msg, **generation_kwargs):
        parsed_input = self._parse_input(msg)
        text_content = parsed_input["text"]
        image = parsed_input["image"]
        
        if hasattr(self.model, 'chat'):
            response, _ = self.model.chat(text_content, image=image, history=None, **self.gen_kwargs, **generation_kwargs)
        else:
            inputs = self.tokenizer(text_content, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    **self.gen_kwargs,
                    **generation_kwargs
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if text_content in response:
                response = response.replace(text_content, "").strip()
        
        return response
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        responses = []
        for msg in sample["messages"]:
            if not self.args.score_target:
                responses.append(self.generate_output(msg))
        ori_sample["response"] = responses
        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()