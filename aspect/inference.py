import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

IMAGE_SIZE = 512
MAX_PIXELS = 1360 * 28 * 28
ADAPTER_SUBFOLDER = "rl_adapter"


def has_adapter(model_id, subfolder):
    if Path(model_id).is_dir():
        return (Path(model_id) / subfolder / "adapter_config.json").exists()
    from huggingface_hub import file_exists
    return file_exists(model_id, f"{subfolder}/adapter_config.json")


def load(model_id, device="cuda", adapter_subfolder=ADAPTER_SUBFOLDER):
    processor = AutoProcessor.from_pretrained(model_id, max_pixels=MAX_PIXELS)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
    if adapter_subfolder and has_adapter(model_id, adapter_subfolder):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_id, subfolder=adapter_subfolder)
    model.eval()
    return model, processor


def prepare_image(image):
    image = image.convert("RGB")
    if max(image.size) > 1024:
        s = 1024 / max(image.size)
        image = image.resize((int(image.size[0] * s), int(image.size[1] * s)))
    return image.resize((IMAGE_SIZE, IMAGE_SIZE))


@torch.inference_mode()
def generate(model, processor, image, question, max_new_tokens=512):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[prepare_image(image)], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Mikezcy/ASPECT-8B")
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    torch.manual_seed(42)
    model, processor = load(args.model)
    print(generate(model, processor, Image.open(args.image), args.question, args.max_new_tokens))


if __name__ == "__main__":
    main()
