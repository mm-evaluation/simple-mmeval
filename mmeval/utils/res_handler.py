import os
import json
import numpy as np
from PIL import Image
from mmeval.utils.sqlitkv import SQLiteKVStore

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class ResponseHandler:
    def __init__(self, args):
        self.rank = args.rank
        self.save_freq = args.save_freq
        self.out_dir = args.out_dir
        self.kvstore = SQLiteKVStore(os.path.join(args.out_dir, f"cache.db"))
        self.load_cache()
        self.pending_result = []  # Pending results for batch saving

    def load_cache(self):
        tmp = self.kvstore.dump_dict()
        self.cache = {int(k): v for k, v in tmp.items()}


    @property
    def length(self):
        return len(self.cache)

    def in_cache(self, id_:str):
        return id_ in self.cache
    
    def check_complete(self, dataset):

        for sample in dataset:
            if not self.in_cache(sample["eval-id"]):
                return False
        print(f"📖 [Shard {self.rank}] Results completed. Saved to {self.out_dir}.")
        return True
        
    def save(self, result:dict):
        assert "eval-id" in result, "eval-id is required"
        
        # Validate message structure: each user prompt must be followed by assistant response
        messages = result.get("messages", [])
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                if i + 1 >= len(messages):
                    raise AssertionError(f"User message at index {i} has no assistant response for sample {result['eval-id']}")
                next_msg = messages[i + 1]
                if next_msg.get("role") != "assistant" or "response" not in next_msg:
                    raise AssertionError(f"User message at index {i} has no assistant response for sample {result['eval-id']}")
        
        # Remove media from each message (contains PIL Image objects)
        for msg in result.get("messages", []):
            if "media" in msg:
                del msg["media"]
        
        # Replace PIL Image objects in sample-level media with placeholder string
        if "media" in result and result["media"]:
            result["media"] = [
                "Image Object" if isinstance(m, Image.Image) else m
                for m in result["media"]
            ]
        
        # Add to in-memory cache
        self.cache[int(result["eval-id"])] = result
        # Add to pending list for batch saving
        self.pending_result.append((str(result["eval-id"]), result))
        # Flush when reaching batch size
        if len(self.pending_result) >= self.save_freq:
            self.flush()

    def flush(self):
        """Batch save all pending results to database."""
        if self.pending_result:
            self.kvstore.batch_put(self.pending_result)
            self.pending_result.clear()
