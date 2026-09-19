"""Submit one representative generation to a comfy backend to trigger profiler capture.
Usage: capture_gen.py [--lora] [--port 8188] [--frames 124]
Without --lora: base model only (guider/scheduler on base, res_multistep) -> plain-base block-loop I/O.
"""
import json, urllib.request, time, os, sys
SC = "/tmp/claude-1000/-mnt-ssdraid-project-flux2/8ab42e25-9cef-47a4-bd14-1e666947fec0/scratchpad"
info = json.load(open(f"{SC}/cut62_info.json"))
prompt = info["prompt"]; REFS = info["refs"]; AUDIOS = info.get("ref_audios", [])
use_lora = "--lora" in sys.argv
port = 8188
for i, a in enumerate(sys.argv):
    if a == "--port": port = int(sys.argv[i+1])
frames = 124
for i, a in enumerate(sys.argv):
    if a == "--frames": frames = int(sys.argv[i+1])
C = f"http://127.0.0.1:{port}"
W, H, LEN, SEED = 1280, 720, frames, 12345
g = {
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
 "3": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
 "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
 "5": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0], "prompt": prompt, "width": W, "height": H, "length": LEN, "ref_image_size": "match"}},
 "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
 "10": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0], "sigmas": ["8", 0], "latent_image": ["5", 1]}},
 "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
 "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
 "13": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "audio": ["12", 0], "fps": 24.0}},
 "14": {"class_type": "SaveVideo", "inputs": {"video": ["13", 0], "filename_prefix": "poc_prof/cap", "format": "auto", "codec": "auto"}},
}
model_ref = ["1", 0]
if use_lora:
    g["lora"] = {"class_type": "MiniMaxH3TurboLoRA", "inputs": {"model": ["1", 0], "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "strength": 1.0, "low_vram": False}}
    model_ref = ["lora", 0]
    g["7"] = {"class_type": "MiniMaxH3TurboSampler", "inputs": {}}
else:
    g["7"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}}
g["8"] = {"class_type": "BasicScheduler", "inputs": {"model": model_ref, "scheduler": "simple", "steps": 4, "denoise": 1.0}}
g["9"] = {"class_type": "BasicGuider", "inputs": {"model": model_ref, "conditioning": ["5", 0]}}
for i, rp in enumerate(REFS[:9]):
    g[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": os.path.basename(rp), "upload": "image"}}
    g["5"]["inputs"][f"ref_images.ref_image_{i}"] = [f"img{i}", 0]
for i, ap in enumerate(AUDIOS[:3]):
    g[f"aud{i}"] = {"class_type": "LoadAudio", "inputs": {"audio": os.path.basename(ap)}}
    g["5"]["inputs"][f"ref_audios.ref_audio_{i}"] = [f"aud{i}", 0]
req = urllib.request.Request(f"{C}/prompt", json.dumps({"prompt": g}).encode(), {"Content-Type": "application/json"})
pid = json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt_id"]
print("submitted", pid, "lora=", use_lora, "frames=", LEN, flush=True)
t0 = time.time()
while True:
    time.sleep(4); h = json.loads(urllib.request.urlopen(f"{C}/history/{pid}", timeout=30).read())
    if pid in h:
        st = h[pid].get("status", {})
        if st.get("completed"): print("DONE %.1fs" % (time.time()-t0)); break
        if any("execution_error" in m[0] for m in st.get("messages", [])):
            print("ERR", [m[1].get("exception_message") for m in st.get("messages", []) if m[0] == "execution_error"][0][:300]); break
