import re
from datasets import load_dataset
from mmeval.data.base import BaseDataset

class MMEvalHFDataset(BaseDataset):
    """Dataset loader for HuggingFace datasets in mm-eval format."""

    def __init__(self, args):
        self.dataset_name = args.dataset.split("@")[1] if "@" in args.dataset else args.dataset
        self.split = args.split
        self.circular = args.circular
        self.resize = args.resize
        if self.resize is not None:
            print(f"Resizing images to {self.resize}x{self.resize}")
        super().__init__(args)

    def _load_raw_data(self, args):
        # Load metadata subset to get jinja_template for the current split
        metadata_ds = load_dataset(self.dataset_name, name="metadata", split=self.split)
        jinja_template = None
        for row in metadata_ds:
            if row["split_name"] == self.split:
                jinja_template = row["jinja_template"]
                break
        if jinja_template is None:
            raise ValueError(f"No jinja_template found for split '{self.split}' in metadata")
        
        # Load default subset with the current split for data
        ds = load_dataset(self.dataset_name, name="default", split=self.split)
        return ds, jinja_template

    def convert_circular(self, **kwargs) -> any:
        """Prepare dataset for circular evaluation."""
        raise NotImplementedError("convert_circular not implemented.")

    def _process_message(self, msg: dict):
        """Process a single message dict, building prompt and processing media."""
        # Use existing prompt if available, otherwise build from template
        if "prompt" in msg:
            prompt = msg["prompt"]
        else:
            prompt = self.build_prompt(self._prompt_template, msg)

        # Normalize to list
        image = msg.get("image", None)
        video = msg.get("video", None)

        if image is None:
            image_list = []
        elif isinstance(image, list):
            image_list = [self.load_image(img) for img in image]
        else:
            image_list = [self.load_image(image)]

        if video is None:
            video_list = []
        elif isinstance(video, list):
            video_list = video
        else:
            video_list = [video]

        media = []
        placeholder_list = re.findall(r"<(video|image)>", prompt)
        for tag in placeholder_list:
            if tag == "image" and image_list:
                img = image_list.pop(0)
                # Resize image if resize parameter is set
                if self.resize is not None:
                    img = self.resize_image(img, self.resize)
                media.append(img)
            elif tag == "video" and video_list:
                media.append(video_list.pop(0))

        return {
            "prompt": prompt,
            "media": media,
            **{k: v for k, v in msg.items() if k not in ("prompt", "image", "video")}
        }

    def _process_sample(self, idx: int):
        sample = dict(self._raw_dataset[idx])
        
        # Check if sample has "messages" key, if not wrap it
        if "messages" in sample:
            messages_raw = sample
        else:
            messages_raw = {"messages": [sample]}
        
        # Process each message
        messages = [self._process_message(msg) for msg in messages_raw["messages"]]

        # Preserve original fields (id, concept_type, etc.) and update with processed data
        sample["eval-id"] = idx
        sample["messages"] = messages
        
        return sample