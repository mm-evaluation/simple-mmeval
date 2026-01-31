# test_llava_multi_image_v2.py
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import numpy as np

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

print("\n" + "="*60)
print("PART 1: Verify source images are different")
print("="*60)
arr1 = np.array(img1)
arr2 = np.array(img2)
print(f"Image 1 shape: {arr1.shape}, mean: {arr1.mean():.2f}")
print(f"Image 2 shape: {arr2.shape}, mean: {arr2.mean():.2f}")
print(f"Source images are different: {not np.array_equal(arr1, arr2)}")

print("\n" + "="*60)
print("PART 2: Build multi-turn conversation")
print("="*60)

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
)

print(f"Turn 1 - pixel_values shape: {inputs_1['pixel_values'].shape}")
print(f"Turn 1 - pixel_values dtype: {inputs_1['pixel_values'].dtype}")
print(f"Turn 1 - BEFORE float16:")
print(f"  pixel_values[0] mean: {inputs_1['pixel_values'][0].float().mean().item():.4f}")
print(f"  pixel_values[0] std: {inputs_1['pixel_values'][0].float().std().item():.4f}")

inputs_1_gpu = inputs_1.to(model.device, torch.float16)
output_1 = model.generate(**inputs_1_gpu, max_new_tokens=50, do_sample=False)
response_1 = processor.batch_decode(output_1[:, inputs_1_gpu["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
print(f"\nTurn 1 Response: {response_1}")

print("\n" + "="*60)
print("PART 3: Multi-turn with second image")
print("="*60)

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
            {"type": "text", "text": "What is in this image?"}
        ]
    }
]

inputs_2 = processor.apply_chat_template(
    conv_history_2, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt"
)

print(f"Turn 2 - pixel_values shape: {inputs_2['pixel_values'].shape}")
print(f"Turn 2 - pixel_values dtype: {inputs_2['pixel_values'].dtype}")

pv = inputs_2['pixel_values']
print(f"\nTurn 2 - BEFORE float16 conversion:")
print(f"  Image 0 mean: {pv[0].float().mean().item():.4f}")
print(f"  Image 0 std:  {pv[0].float().std().item():.4f}")
print(f"  Image 1 mean: {pv[1].float().mean().item():.4f}")
print(f"  Image 1 std:  {pv[1].float().std().item():.4f}")

is_equal = torch.allclose(pv[0].float(), pv[1].float(), atol=1e-5)
print(f"\n  Images are EQUAL (allclose): {is_equal}")
print(f"  Images are DIFFERENT: {not is_equal}")

print(f"\n  First 5 values of Image 0: {pv[0].flatten()[:5].tolist()}")
print(f"  First 5 values of Image 1: {pv[1].flatten()[:5].tolist()}")

inputs_2_gpu = inputs_2.to(model.device, torch.float16)
pv_gpu = inputs_2_gpu['pixel_values']
print(f"\nTurn 2 - AFTER float16 conversion:")
print(f"  Image 0 sum: {pv_gpu[0].sum().item()}")
print(f"  Image 1 sum: {pv_gpu[1].sum().item()}")

output_2 = model.generate(**inputs_2_gpu, max_new_tokens=50, do_sample=False)
response_2 = processor.batch_decode(output_2[:, inputs_2_gpu["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
print(f"\nTurn 2 Response: {response_2}")

print("\n" + "="*60)
print("PART 4: Control test - boat image alone")
print("="*60)

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
print(f"Control (boat only) Response: {response_boat}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Turn 1 (truck):              {response_1}")
print(f"Turn 2 (boat in multi-turn): {response_2}")
print(f"Control (boat alone):        {response_boat}")
print()
if "boat" in response_2.lower() or "ship" in response_2.lower() or "water" in response_2.lower():
    print("✅ Multi-turn multi-image appears to be WORKING")
else:
    print("❌ Multi-turn multi-image has ISSUES - model not recognizing second image correctly")
print("="*60)

