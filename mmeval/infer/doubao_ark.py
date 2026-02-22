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
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs, filter_gen_kwargs

load_dotenv()


def encode_image(image):
    """Encode PIL Image to base64"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def encode_video(video_path):
    """Encode video file to base64"""
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.default_model_kwargs = {}
        self.default_gen_kwargs = {}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)
        
        super().__init__(args)
        
    def load_model(self, args):
        api_key = os.getenv("DOUBAO_API_KEY")
        if not api_key:
            raise ValueError("DOUBAO_API_KEY not found in environment variables")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            **self.model_kwargs
        )
        self.model_name = args.model_name_or_path.split("/")[-1]
        self.gen_kwargs = filter_gen_kwargs(self.gen_kwargs, self.client.chat.completions.create)

    def parse_input(self, message: dict):
        question = message["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        media_list = message.get('media', [])
        content = []
        media_idx = 0

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                image = media_list[media_idx]
                media_idx += 1
                base64_image = encode_image(image)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpg;base64,{base64_image}"
                    }
                })
            elif chunk == constants.video:
                video_path = media_list[media_idx]
                media_idx += 1
                base64_video = encode_video(video_path)
                content.append({
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:video/mp4;base64,{base64_video}"
                    }
                })
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
        message = sample["messages"][0]
        ori_sample = copy.deepcopy(sample)
        content = self.parse_input(message)

        if not self.args.score_target:
            response = self._generate_response(content)
            ori_sample["messages"].append({"role": "assistant", "response": response})
        else:
            raise NotImplementedError("Score target mode not supported for Doubao API models")

        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()

