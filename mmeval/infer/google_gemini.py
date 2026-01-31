import re
import copy
import os
import time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs

load_dotenv()


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.default_gen_kwargs = {}
        self.default_model_kwargs = {}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)
        
        super().__init__(args)
        
    def load_model(self, args):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key, **self.model_kwargs)
        self.client = genai
        self.model_name = args.model_name_or_path.split("/")[-1]
        self.model = self.client.GenerativeModel(self.model_name)
        self.generation_config = genai.types.GenerationConfig(**self.gen_kwargs)

    def _wait_for_file_active(self, video_file, timeout=120, poll_interval=2):
        start_time = time.time()
        file_name = video_file.name
        
        while time.time() - start_time < timeout:
            file = self.client.get_file(file_name)
            
            if file.state.name == "ACTIVE":
                return file
            elif file.state.name == "FAILED":
                raise RuntimeError(f"Video file processing failed: {file_name}")
            
            # Still processing, wait before checking again
            time.sleep(poll_interval)
        
        # Timeout reached
        raise TimeoutError(
            f"Video file {file_name} did not become ACTIVE within {timeout} seconds. "
            f"Current state: {file.state.name}"
        )

    def parse_input(self, msg):
        question = msg["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        media_list = copy.deepcopy(msg["media"])

        contents = []

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                image = media_list.pop(0)
                contents.append(image)
            elif chunk == constants.video:
                video_path = media_list.pop(0)
                print(f"Uploading video: {video_path}")
                video_file = self.client.upload_file(path=video_path)
                print(f"Waiting for video file to become ACTIVE: {video_file.name}")
                video_file = self._wait_for_file_active(video_file)
                print(f"Video file is ready: {video_file.name}")
                contents.append(video_file)
            else:
                contents.append(chunk)

        return contents

    def _generate_response(self, contents):
        response = self.model.generate_content(
            contents,
            generation_config=self.generation_config
        )
        
        return response.text

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        responses = []
        for msg in sample["messages"]:
            contents = self.parse_input(msg)

            if not self.args.score_target:
                responses.append(self._generate_response(contents))
            else:
                raise NotImplementedError("Score target mode not supported for Google Gemini API models")

        ori_sample["response"] = responses
        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()

