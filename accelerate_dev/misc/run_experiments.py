#!/usr/bin/env python3
import subprocess
import re
import os
import json
import time

precisions = ["fp32", "tf32", "fp16"]

# Two configurations
configs = [
    {
        "name": "default",
        "e_cfg_scale": "1",
        "emo": None,
        "res_path_pattern": "./experiment/tensorrt_res/output_float-{fmt}_dec-{dec}/default"
    },
    {
        "name": "emo=sad",
        "e_cfg_scale": "10",
        "emo": "sad",
        "res_path_pattern": "./experiment/tensorrt_res/output_float-{fmt}_dec-{dec}/emo=sad"
    }
]

results = []

# Regular expressions to parse the required metrics
metrics_regex = {
    "sampling_time": re.compile(r"\[#TENSORRT\]>\s+Sampling completed in\s+([\d\.]+)\s+seconds"),
    "decoding_time": re.compile(r"\[#TENSORRT\]>\s+Decoding completed in\s+([\d\.]+)\s+seconds"),
    "decoding_fps": re.compile(r"\[#TENSORRT\]>\s+Achieved FPS =\s+([\d\.]+)\s+frames/sec"),
    "inference_time": re.compile(r">\s+\[#TENSORRT\]\s+Inference completed\s+\(\)\s+in\s+([\d\.]+)\s+seconds"),
    "inference_fps": re.compile(r">\s+\[#TENSORRT\]\s+Inference FPS =\s+([\d\.]+)\s+frames/sec"),
    "total_time": re.compile(r">\s+\[#TENSORRT\]\s+Total execution\s+\(Preprocess\s+\+\s+TENSORRT\s+\+\s+Save\s*\)\s+time:\s+([\d\.]+)\s+seconds"),
    "total_fps": re.compile(r">\s+\[#TENSORRT\]\s+Total execution\s+FPS\s+=\s+([\d\.]+)\s+frames/sec")
}

total_runs = len(precisions) * len(precisions) * len(configs)
current_run = 0

for fmt in precisions:
    for dec in precisions:
        for config in configs:
            current_run += 1
            print("="*80)
            print(f"RUN {current_run}/{total_runs}: FMT={fmt}, DEC={dec}, CONFIG={config['name']}")
            print("="*80)
            
            fmt_path = f"trt_models/fmt_onnx_maskfill_addcfg_{fmt}.trt"
            dec_path = f"trt_models/float_decoder_{dec}.trt"
            res_path = config["res_path_pattern"].format(fmt=fmt, dec=dec)
            
            cmd = [
                "python", "generate_with_tensorrt.py",
                "--ref_path", "./extra_assests/face/Ton.png",
                "--aud_path", "assets/aud-sample-vs-1.wav",
                "--seed", "15",
                "--a_cfg_scale", "2",
                "--e_cfg_scale", config["e_cfg_scale"],
                "--trt_model_path", fmt_path,
                "--trt_decoder_path", dec_path,
                "--res_video_path", res_path
            ]
            if config["emo"]:
                cmd.extend(["--emo", config["emo"]])
                
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "1"
            
            print(f"Executing: {' '.join(cmd)}")
            
            start_time = time.time()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
            
            stdout_lines = []
            for line in proc.stdout:
                print(line, end="")
                stdout_lines.append(line)
                
            proc.wait()
            elapsed = time.time() - start_time
            print(f"Process finished with code {proc.returncode} in {elapsed:.2f} seconds.")
            
            # Parse output
            stdout_text = "".join(stdout_lines)
            run_metrics = {
                "fmt": fmt,
                "dec": dec,
                "config": config["name"],
                "success": proc.returncode == 0
            }
            
            for key, rx in metrics_regex.items():
                m = rx.search(stdout_text)
                if m:
                    run_metrics[key] = float(m.group(1))
                else:
                    run_metrics[key] = None
                    
            results.append(run_metrics)

# Save raw results as JSON
os.makedirs("accelerate_dev", exist_ok=True)
with open("accelerate_dev/experiment_results_new.json", "w") as f:
    json.dump(results, f, indent=4)

# Create markdown report
def build_markdown_table(data, config_name):
    filtered = [r for r in data if r["config"] == config_name]
    md = f"### Configuration: {config_name}\n\n"
    md += "| FMT Precision | Decoder Precision | Sampling Time (s) | Decoding Time (s) | Decoding FPS | Inference Time (s) | Inference FPS | Total Time (s) | Total FPS | Success |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for r in filtered:
        success_str = "✅" if r["success"] else "❌"
        sampling = f"{r['sampling_time']:.2f}" if r['sampling_time'] is not None else "N/A"
        decoding = f"{r['decoding_time']:.2f}" if r['decoding_time'] is not None else "N/A"
        dec_fps = f"{r['decoding_fps']:.2f}" if r['decoding_fps'] is not None else "N/A"
        inf_time = f"{r['inference_time']:.2f}" if r['inference_time'] is not None else "N/A"
        inf_fps = f"{r['inference_fps']:.2f}" if r['inference_fps'] is not None else "N/A"
        tot_time = f"{r['total_time']:.2f}" if r['total_time'] is not None else "N/A"
        tot_fps = f"{r['total_fps']:.2f}" if r['total_fps'] is not None else "N/A"
        md += f"| {r['fmt'].upper()} | {r['dec'].upper()} | {sampling} | {decoding} | {dec_fps} | {inf_time} | {inf_fps} | {tot_time} | {tot_fps} | {success_str} |\n"
    return md

report = "# TensorRT Precision Combinations Benchmark Report\n\n"
report += "This report summarizes the performance metrics of all 9 precision combinations (FMT vs Decoder) for both the default configuration and the expressive emotion configuration.\n\n"
report += build_markdown_table(results, "default")
report += "\n"
report += build_markdown_table(results, "emo=sad")

print("\n" + "#"*40 + "\nREPORT GENERATED\n" + "#"*40 + "\n")
print(report)

# Write report to markdown file in accelerate_dev/
with open("accelerate_dev/experiment_results.md", "w") as f:
    f.write(report)

# # Write to the artifacts directory too
# artifact_dir = "/home/mint/.gemini/antigravity-cli/brain/2fe2197c-9a5a-42d2-96f7-75abbfc62769"
# if os.path.exists(artifact_dir):
#     with open(os.path.join(artifact_dir, "experiment_results.md"), "w") as f:
#         f.write(report)
