import re
import copy
import torch
import os
import uuid
from PIL import Image
import transformers
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from qwen_vl_utils import process_vision_info

from mmeval.infer.task import Task
from mmeval.utils import spec_tokens 
from mmeval.utils.argparser import ModelArguments, DataArguments, InferenceArguments

class TaskRunner(Task):
    def __init__(self, model_arguments, data_arguments, inference_arguments):
        super().__init__(model_arguments, data_arguments, inference_arguments)
        
    
    def load_model(self, args):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name_or_path, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    def run_sample(self, sample:dict):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ori_sample = copy.deepcopy(sample)
        temp_files_to_delete = []  # Track temp files created for this sample
        
        # Handle media_path: if not exists, convert PIL objects to temp files
        if 'media_path' not in sample and 'media' in sample:
            sample = self._prepare_media_paths(sample)
            # Track which files are temporary (created by us)
            temp_files_to_delete = [path for path in sample['media_path'] 
                                    if path.startswith("/MLLM_Eval/data/temp_img/temp_")]
        
        messages = self.parse_input(sample)
        
    
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        ori_sample["response"] = output_text
        ori_sample.pop("media", None)
        ori_sample.pop("media_path", None)
        
        # Clean up temporary files after inference
        for temp_file in temp_files_to_delete:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                # Log error but don't fail the inference
                print(f"Warning: Failed to delete temporary file {temp_file}: {e}")
        
        return ori_sample

    def _prepare_media_paths(self, sample):
        """Convert PIL objects from media field to temporary file paths."""
        new_sample = copy.deepcopy(sample)
        temp_dir = "/MLLM_Eval/data/temp_img"
        os.makedirs(temp_dir, exist_ok=True)
        
        media_paths = []
        for i, media_obj in enumerate(sample['media']):
            if isinstance(media_obj, Image.Image):
                # Generate unique filename
                unique_id = str(uuid.uuid4())[:8]
                temp_path = os.path.join(temp_dir, f"temp_{unique_id}_{i}.png")
                media_obj.save(temp_path)
                media_paths.append(temp_path)
            else:
                # If it's already a path, use it as is
                media_paths.append(media_obj)
        
        new_sample['media_path'] = media_paths
        return new_sample

    def parse_input(self, sample:dict):
        question = sample["prompt"]
        # placeholder <>, can be image, video, audio, etc.
        q_chunks = re.split(r'(<[^>]*>)', question)
        images = copy.deepcopy(sample['media_path'])

        messages = [
            {
                "role": "user",
                "content": []
            }
        ]

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            
            if any(p in chunk for p in spec_tokens.all):
                # TODO: Qwen2.5-VL might support other modality
                assert chunk == spec_tokens.image, f"Unsupported placeholder {chunk}"

                media_file = images.pop(0)
                messages[0]["content"].append(
                    {
                        "type": "image",
                        "image": media_file
                    }
                )       

            else:
                messages[0]["content"].append(
                    {
                        "type": "text",
                        "text": chunk
                    }
                )
        return messages

    

if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, InferenceArguments))
    model_arguments, data_arguments, inference_arguments = parser.parse_args_into_dataclasses()
    

    model_evaluator = TaskRunner(model_arguments, data_arguments, inference_arguments)

    model_evaluator.inference_dataset()

        
