import json
from datasets import load_dataset, get_dataset_config_names
from mmeval.data.base import BaseDataset


class MMEvalHFDataset(BaseDataset):
    """Dataset loader for HuggingFace datasets in mm-eval format."""

    def __init__(self, args):
        dataset_str = args.dataset.split("@")[1] if "@" in args.dataset else args.dataset
        # Support optional config suffix: "mm-eval/MMMU_Pro:vision"
        if ":" in dataset_str:
            self.dataset_name, self.dataset_config = dataset_str.rsplit(":", 1)
        else:
            self.dataset_name = dataset_str
            self.dataset_config = None
        self.split = args.split
        self.circular = args.circular
        self.resize = args.resize  # Used by base._process_messages for image resizing
        if self.resize is not None:
            print(f"Resizing images to {self.resize}x{self.resize}")
        super().__init__(args)

    def _load_raw_data(self, args) -> tuple:
        all_configs = get_dataset_config_names(self.dataset_name)
        if self.dataset_config:
            data_config = self.dataset_config
            metadata_config = f"{data_config}_metadata"
        else:
            # Auto-detect: older datasets use "metadata"/"default"; newer ones use "<name>_metadata"
            if "metadata" in all_configs:
                data_config = "default"
                metadata_config = "metadata"
            else:
                data_configs = [c for c in all_configs if not c.endswith("_metadata")]
                if not data_configs:
                    raise ValueError(
                        f"No data configs found for {self.dataset_name}. Available: {all_configs}\n"
                        f"Specify a config explicitly, e.g. mmeval_hf@{self.dataset_name}:<config_name>"
                    )
                data_config = data_configs[0]
                metadata_config = f"{data_config}_metadata"
                print(f"Auto-selected config '{data_config}' (available: {all_configs}). "
                      f"Use mmeval_hf@{self.dataset_name}:<config_name> to pick a specific one.")

        # Some upstream HF datasets (e.g. mm-eval/MMMU, mm-eval/MMMU-pro) ship only
        # the data config without a paired '<config>_metadata'. Others (e.g.
        # mm-eval/DynaMath) keep the metadata config name but ship it empty for
        # the requested split (load_dataset raises 'Instruction "<split>"
        # corresponds to no data!'). Treat the metadata split as optional.
        # verification_mode='no_checks' tolerates upstream split-size drift
        # (e.g. mm-eval/DynaMath data config's dataset_infos.json lists 1 example
        # while the parquet actually has 501).
        dataset_template = None
        if metadata_config in all_configs:
            try:
                metadata_ds = load_dataset(
                    self.dataset_name, name=metadata_config, split=self.split,
                    verification_mode="no_checks",
                )
                if len(metadata_ds) > 0:
                    dataset_template = metadata_ds[0]["jinja_template"]
            except Exception as e:
                print(f"[mmeval_hf] metadata config '{metadata_config}' present but unusable "
                      f"({type(e).__name__}: {str(e)[:120]}); proceeding with dataset_template=None")

        ds = load_dataset(
            self.dataset_name, name=data_config, split=self.split,
            verification_mode="no_checks",
        )
        return ds, dataset_template  # Return HF's jinja_template as dataset template

    def convert_circular(self, **kwargs) -> any:
        """Prepare dataset for circular evaluation."""
        raise NotImplementedError("convert_circular not implemented.")

    def _process_sample(self, idx: int):
        sample = dict(self._raw_dataset[idx])
        
        # Add eval-id if not present
        if "eval-id" not in sample:
            sample["eval-id"] = idx
        
        messages = sample["messages"]
        message_list = json.loads(messages) if isinstance(messages, str) else messages
        
        # Get media from sample level (HF datasets may store as single image or list)
        media = sample.get("media")
        if media is None:
            media_list = []
        elif isinstance(media, list):
            media_list = media
        else:
            media_list = [media]
        
        # Ensure sample["media"] is always a list
        sample["media"] = media_list

        # If the default template will be used (no per-dataset jinja template)
        # and the user message's question text does not contain media placeholders
        # while media is present (e.g. mm-eval/DynaMath, whose 'metadata' config
        # is unusable upstream), prepend "<image>" tokens to the question so the
        # default template renders them into the prompt. Without this the
        # framework raises "Prompt/media mismatch: used_media=0, total_media=N".
        if self._dataset_template is None and media_list:
            for msg in message_list:
                if not isinstance(msg, dict) or msg.get("role") != "user":
                    continue
                q = msg.get("question")
                if not isinstance(q, str):
                    continue
                # Check for existing placeholders across ALL fields the default
                # template renders (question + options values + hint), not just the
                # question. Some datasets (e.g. mm-eval/MMMU) reference images only
                # in the options; checking the question alone would wrongly conclude
                # "no placeholder" and over-inject, producing placeholders > media.
                rendered_fields = [q]
                opts = msg.get("options")
                if isinstance(opts, dict):
                    rendered_fields += [str(v) for v in opts.values()]
                elif isinstance(opts, (list, tuple)):
                    rendered_fields += [str(v) for v in opts]
                hint = msg.get("hint")
                if isinstance(hint, str):
                    rendered_fields.append(hint)
                if any(("<image>" in f or "<video>" in f) for f in rendered_fields):
                    break  # at least one placeholder present; trust it
                # Prepend one "<image>" per media item (only image media is auto-injected).
                msg["question"] = ("<image>\n" * len(media_list)) + q
                break

        # Process all messages with sample-level media indexed by placeholder order
        sample["messages"] = self._process_messages(message_list, media_list)

        return sample
