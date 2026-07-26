import json

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from mmeval.data.base import BaseDataset


class MMEvalHFDataset(BaseDataset):
    """Dataset loader for HuggingFace datasets in mm-eval format.

    Reads the subset manifest from the dataset repo's metadata.json and injects
    the scoring-relevant metadata as the flat top-level `dataset_meta` sample
    field. Rows are expected in the flat layout (grading fields as top-level
    columns; messages[0] carries render inputs only) — non-conforming rows are
    passed through unchanged and surface at scoring time via the scorer's
    explicit layout rejection.
    """

    def __init__(self, args):
        self.dataset_name = args.dataset.split("@", 1)[1]
        self.subset = getattr(args, "subset", None)
        self.split = args.split
        self.circular = args.circular
        self.resize = args.resize  # Used by base._process_messages for image resizing
        if self.resize is not None:
            print(f"Resizing images to {self.resize}x{self.resize}")
        super().__init__(args)

    def _load_raw_data(self, args) -> tuple:
        meta_path = hf_hub_download(
            repo_id=self.dataset_name,
            filename="metadata.json",
            repo_type="dataset",
        )
        with open(meta_path, encoding="utf-8") as f:
            subsets = json.load(f).get("subsets") or {}
        if not subsets:
            raise ValueError(f"{self.dataset_name} has no subsets")
        if self.subset is None:
            if len(subsets) != 1:
                raise ValueError(
                    f"{self.dataset_name} has {len(subsets)} subsets "
                    f"({list(subsets)}); pass --subset to pick one"
                )
            self.subset = next(iter(subsets))
        elif self.subset not in subsets:
            raise ValueError(
                f"{self.dataset_name}: subset {self.subset!r} not in {list(subsets)}"
            )
        subset_block = subsets[self.subset]
        dataset_template = subset_block.get("prompt_template")

        # Scoring data-flow contract: inject the subset's scoring-relevant
        # metadata into every sample as the flat top-level `dataset_meta`
        # field, so it lands in result.json and the scorer can resolve the
        # official grading protocol without access to the HF manifest. CLI
        # flags on the scorer always override these values.
        if "score_type" in subset_block:
            raise ValueError(
                f"{self.dataset_name} subset {self.subset!r}: metadata.json "
                f"contains an unsupported `score_type` key. Declare the protocol "
                f"with the `score_pipeline` schema instead (docs/en/SCORING.md, "
                f"'Dataset metadata contract'), then re-run inference."
            )
        # Dataset identity always travels in result.json, even when no scoring
        # protocol is declared, so the scorer's resume fingerprint can tell
        # apart caches produced for a different split/subset/dataset scored
        # into the same out_dir (eval-ids 0..N would otherwise collide).
        meta_block = {
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "split": self.split,
        }
        for key in ("task_type", "score_params"):
            value = subset_block.get(key)
            if value:
                meta_block[key] = value
        if "score_pipeline" in subset_block:
            # Declared protocol passes through verbatim; [] (explicitly no
            # official protocol) is a meaningful value, so test presence.
            meta_block["score_pipeline"] = subset_block["score_pipeline"]
        note = ((subset_block.get("score_protocol") or {}).get("note") or "").strip()
        if note:
            meta_block["score_note"] = note
        self._dataset_meta = meta_block

        # Prefer multi-config layout: HF config name == mm-eval subset name.
        # Fall back to the single-`default` config layout, where one config
        # holds all subsets' splits (named `<subset>_<split>`).
        try:
            ds = load_dataset(self.dataset_name, name=self.subset, split=self.split)
        except (ValueError, FileNotFoundError):
            ds = load_dataset(self.dataset_name, name="default", split=self.split)
        return ds, dataset_template

    def convert_circular(self, **kwargs) -> any:
        """Prepare dataset for circular evaluation."""
        raise NotImplementedError("convert_circular not implemented.")

    def _process_sample(self, idx: int):
        sample = dict(self._raw_dataset[idx])

        if "eval-id" not in sample:
            sample["eval-id"] = idx

        messages = sample["messages"]
        message_list = json.loads(messages) if isinstance(messages, str) else messages

        media = sample.get("media")
        if media is None:
            media_list = []
        elif isinstance(media, list):
            media_list = media
        else:
            media_list = [media]

        sample["media"] = media_list
        sample["messages"] = self._process_messages(message_list, media_list)

        if self._dataset_meta:
            sample.setdefault("dataset_meta", dict(self._dataset_meta))

        return sample
