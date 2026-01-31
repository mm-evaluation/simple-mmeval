# test_llava_multi_image.py
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

print("Loading model...")
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    device_map="auto",
    low_cpu_mem_usage=True
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
print("Model loaded!")

img1_path = "test_bed/modality_test/media/448/truck.png"
img2_path = "test_bed/modality_test/media/448/boat.png"
img1 = Image.open(img1_path)
img2 = Image.open(img2_path)

# === Turn 1: Ask about truck ===
print("\n=== Turn 1: Ask about truck ===")
conv_history_1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img1},
            {"type": "text", "text": "What is in the image?"}
        ]
    }
]

inputs_1 = processor.apply_chat_template(
    conv_history_1, tokenize=True, add_generation_prompt=True, 
    return_dict=True, return_tensors="pt"
).to(model.device, torch.float16)

print(f"Turn 1 - pixel_values shape: {inputs_1['pixel_values'].shape}")

output_1 = model.generate(**inputs_1, max_new_tokens=50, do_sample=False)
response_1 = processor.batch_decode(output_1[:, inputs_1["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
print(f"Response 1: {response_1}")

# === Turn 2: Ask about boat ===
print("\n=== Turn 2: Ask about boat ===")
conv_history_2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img1},
            {"type": "text", "text": "What is in the image?"}
        ]
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": response_1}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img2},
            {"type": "text", "text": "What is in the image?"}
        ]
    }
]

inputs_2 = processor.apply_chat_template(
    conv_history_2, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt"
).to(model.device, torch.float16)

print(f"Turn 2 - pixel_values shape: {inputs_2['pixel_values'].shape}")
print(f"Expected: [2, x, 3, 384, 384] (2 images)")

# Verify images are different
print(f"\nImage 1 pixel sum: {torch.sum(inputs_2['pixel_values'][0])}")
print(f"Image 2 pixel sum: {torch.sum(inputs_2['pixel_values'][1])}")
print(f"Images are different: {torch.sum(inputs_2['pixel_values'][0]) != torch.sum(inputs_2['pixel_values'][1])}")

output_2 = model.generate(**inputs_2, max_new_tokens=50, do_sample=False)
response_2 = processor.batch_decode(output_2[:, inputs_2["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
print(f"Response 2: {response_2}")

# === Control: Ask about boat directly ===
print("\n=== Control: Ask about boat directly ===")
conv_boat_only = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img2},
            {"type": "text", "text": "What is in the image?"}
        ]
    }
]

inputs_boat = processor.apply_chat_template(
    conv_boat_only, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt"
).to(model.device, torch.float16)

output_boat = model.generate(**inputs_boat, max_new_tokens=50, do_sample=False)
response_boat = processor.batch_decode(output_boat[:, inputs_boat["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
print(f"Response (boat only): {response_boat}")

print("\n" + "="*60)
print("SUMMARY:")
print(f"  Turn 1 (truck): {response_1}")
print(f"  Turn 2 (boat in multi-turn): {response_2}")
print(f"  Control (boat alone): {response_boat}")
print("="*60)
