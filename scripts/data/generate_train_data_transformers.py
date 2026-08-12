import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate JSONL conversations through Muse-Glimmer directly "
        "with transformers. Alternative to the sglang-based "
        "generate_train_data.py (useful when no sglang server is available)."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-file-path", required=True)
    parser.add_argument("--output-file-path", required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def count_lines(path):
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def find_resume_offset(output_path, error_path):
    if not os.path.exists(output_path):
        return 0, 0, 0

    success_count = count_lines(output_path)
    error_count = count_lines(error_path) if os.path.exists(error_path) else 0
    return success_count + error_count, success_count, error_count


def main():
    args = parse_args()
    total_lines = count_lines(args.input_file_path)
    error_path = args.output_file_path.replace(".jsonl", "_error.jsonl")
    skip_lines, existing_success, existing_errors = (
        find_resume_offset(args.output_file_path, error_path)
        if args.resume
        else (0, 0, 0)
    )
    if skip_lines >= total_lines:
        print(f"All {total_lines} samples are already processed.")
        return
    if args.resume and skip_lines > 0:
        print(
            "Resume mode: "
            f"{existing_success} success, {existing_errors} errors, skip {skip_lines}"
        )

    print("Loading model and processor...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    file_mode = "a" if args.resume and skip_lines > 0 else "w"
    stats = {"success": 0, "errors": 0}
    progress_total = (
        total_lines
        if args.num_samples is None
        else min(total_lines, skip_lines + args.num_samples)
    )

    with (
        open(args.input_file_path, "r", encoding="utf-8") as input_handle,
        open(args.output_file_path, file_mode, encoding="utf-8") as output_handle,
        open(error_path, file_mode, encoding="utf-8") as error_handle,
    ):
        for _ in range(skip_lines):
            next(input_handle, None)

        progress = tqdm(total=progress_total, initial=skip_lines, desc="Processing")
        submitted = 0
        for line in input_handle:
            if args.num_samples is not None and submitted >= args.num_samples:
                break
            submitted += 1

            sample = json.loads(line)
            conversations = sample.get("conversations")
            if not conversations:
                sample["status"] = "error"
                sample["error"] = "Missing conversations"
                error_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats["errors"] += 1
                progress.update(1)
                continue
            if conversations[0].get("role") == "assistant":
                sample["status"] = "error"
                sample["error"] = "Data starts with an assistant message"
                error_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats["errors"] += 1
                progress.update(1)
                continue

            regenerated = []
            failed = False
            for message in conversations:
                role = message.get("role")
                if role == "system":
                    regenerated.append(message)
                    continue
                if role == "assistant":
                    continue
                if role != "user":
                    sample["status"] = "error"
                    sample["error"] = f"Invalid message role: {role}"
                    error_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    stats["errors"] += 1
                    failed = True
                    break

                regenerated.append(message)
                try:
                    inputs = processor.apply_chat_template(
                        list(regenerated),
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=args.max_tokens,
                            do_sample=args.temperature > 0.0,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                        )
                    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
                    content = processor.decode(
                        new_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    regenerated.append({"role": "assistant", "content": content})
                except Exception as exc:
                    sample["status"] = "error"
                    sample["error"] = str(exc)
                    error_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    stats["errors"] += 1
                    failed = True
                    break
                progress.update(1)

            if not failed:
                sample["conversations"] = regenerated
                sample["status"] = "success"
                output_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats["success"] += 1

    progress.close()
    print("Processing completed.")
    print(f"  success: {stats['success']}")
    print(f"  errors: {stats['errors']}")


if __name__ == "__main__":
    main()
