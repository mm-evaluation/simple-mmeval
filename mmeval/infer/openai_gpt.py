import re
import copy
import os
import base64
import io

from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
 
load_dotenv()


def encode_image(image):
    """Encode PIL Image to base64"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.default_gen_kwargs = {}
        self.default_model_kwargs = {}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)
        
        super().__init__(args)
        
    def load_model(self, args):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key, **self.model_kwargs)
        self.model_name = args.model_name_or_path.split("/")[-1]

    def parse_input(self, msg):
        question = msg["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        media_list = copy.deepcopy(msg["media"])

        content = []

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                image = media_list.pop(0)
                base64_image = encode_image(image)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            elif chunk == constants.video:
                raise NotImplementedError("OpenAI GPT does not support video input")
            else:
                content.append({
                    "type": "text",
                    "text": chunk
                })

        return content

    def _generate_response(self, content):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            **self.gen_kwargs
        )
        
        return response.choices[0].message.content

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        responses = []
        for msg in sample["messages"]:
            content = self.parse_input(msg)

            if not self.args.score_target:
                responses.append(self._generate_response(content))
            else:
                raise NotImplementedError("Score target mode not supported for OpenAI API models")

        ori_sample["response"] = responses
        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()

