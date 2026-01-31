import os
import re
import json
from PIL import Image

from mmeval.data.base import BaseDataset


class LocalJSONDataset(BaseDataset):
    """Dataset class for loading local JSON files."""
    
    def __init__(self, args):
        self.data_file = args.infile
        self.img_dir = args.img_dir
        self.template_arg = args.template
        super().__init__(args)

    def _load_template(self, template_arg):
        """Load template from file path or use string directly."""
        if template_arg is None:
            return None
        # Check if it's a file path
        if os.path.exists(template_arg):
            with open(template_arg, "r") as f:
                return f.read()
        # Otherwise treat as template string directly
        return template_arg

    def _load_raw_data(self, args):
        data = json.load(open(self.data_file, "r"))
        for i, sample in enumerate(data):
            assert "eval-id" not in sample, "eval-id already exists"
            sample["eval-id"] = i
        template = self._load_template(self.template_arg)
        return data, template

    def _process_message(self, msg: dict):
        """Process a single message dict, building prompt and processing media."""
        msg = dict(msg)
        
        # Use existing prompt if available, otherwise build from template
        if "prompt" in msg:
            prompt = msg["prompt"]
        elif self._prompt_template is not None:
            prompt = self.build_prompt(self._prompt_template, msg)
        else:
            raise ValueError("No prompt found and no template provided")
        
        # Join media paths with img_dir
        media = [os.path.join(self.img_dir, f) for f in msg.get("media", [])]
        
        placeholder_list = re.findall(r"<(?:video|image)>", prompt)
        assert len(placeholder_list) == len(media), \
            f"Number of media placeholders ({len(placeholder_list)}) does not match number of media files ({len(media)})"
        
        msg["media"] = [
            self.load_image(f) if placeholder == "<image>" else f
            for placeholder, f in zip(placeholder_list, media)
        ]
        msg["prompt"] = prompt

        return msg
    
    def _process_sample(self, idx: int):
        sample = dict(self._raw_dataset[idx])
        eval_id = sample.pop("eval-id")
        
        # Check if sample has "messages" key, if not wrap it
        if "messages" in sample:
            messages_raw = sample
        else:
            messages_raw = {"messages": [sample]}
        
        # Process each message
        messages = [self._process_message(msg) for msg in messages_raw["messages"]]

        # Preserve original fields (id, concept_type, etc.) and update with processed data
        sample["eval-id"] = eval_id
        sample["messages"] = messages
        
        return sample

    def __repr__(self):
        if self.parallel_per_task > 1:
            return f"local@{self.data_file.split('/')[-1]}(rank={self.rank}/{self.parallel_per_task}, local={len(self)}, global={self.global_length})"
        else:
            return f"local@{self.data_file.split('/')[-1]}(samples={len(self)})"
    
    def __str__(self):
        return self.__repr__()
