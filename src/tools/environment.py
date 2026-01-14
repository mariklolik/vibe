"""Environment setup tools - create envs, install deps, setup docker."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

DOCKERFILE_TEMPLATE = """# Auto-generated Dockerfile for research experiments
FROM {base_image}

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    curl \\
    vim \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY {requirements_file} /workspace/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /workspace/

# Default command
CMD ["bash"]
"""


async def create_experiment_env(
    name: str,
    python: str = "3.10",
    use_conda: bool = False,
) -> str:
    if use_conda:
        conda_path = shutil.which("conda") or shutil.which("mamba")
        
        if not conda_path:
            return json.dumps({
                "success": False,
                "error": "conda/mamba not found. Install Anaconda or Miniconda.",
                "alternative": "Set use_conda=false to use venv instead",
            })
        
        cmd = [conda_path, "create", "-n", name, f"python={python}", "-y"]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            if result.returncode == 0:
                return json.dumps({
                    "success": True,
                    "env_name": name,
                    "python": python,
                    "type": "conda",
                    "activate": f"conda activate {name}",
                })
            else:
                if "already exists" in result.stderr:
                    return json.dumps({
                        "success": True,
                        "env_name": name,
                        "message": "Environment already exists",
                        "activate": f"conda activate {name}",
                    })
                return json.dumps({
                    "success": False,
                    "error": result.stderr,
                })
        except subprocess.TimeoutExpired:
            return json.dumps({
                "success": False,
                "error": "Environment creation timed out",
            })
    else:
        env_path = Path(f"./.venvs/{name}")
        env_path.parent.mkdir(parents=True, exist_ok=True)
        
        python_cmd = f"python{python}" if python else "python3"
        
        try:
            result = subprocess.run(
                [python_cmd, "-m", "venv", str(env_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode == 0:
                return json.dumps({
                    "success": True,
                    "env_name": name,
                    "path": str(env_path),
                    "type": "venv",
                    "activate": f"source {env_path}/bin/activate",
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": result.stderr or "Failed to create venv",
                })
        except FileNotFoundError:
            return json.dumps({
                "success": False,
                "error": f"Python {python} not found",
            })


async def install_dependencies(
    env_name: str,
    requirements: Optional[list[str]] = None,
    requirements_file: Optional[str] = None,
) -> str:
    conda_path = shutil.which("conda") or shutil.which("mamba")
    
    if requirements_file:
        req_path = Path(requirements_file)
        if not req_path.exists():
            return json.dumps({
                "success": False,
                "error": f"Requirements file not found: {requirements_file}",
            })
        
        if conda_path:
            cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {env_name} && pip install -r {requirements_file}"
        else:
            venv_path = Path(f"./.venvs/{env_name}")
            cmd = f"source {venv_path}/bin/activate && pip install -r {requirements_file}"
    elif requirements:
        packages = " ".join(requirements)
        if conda_path:
            cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {env_name} && pip install {packages}"
        else:
            venv_path = Path(f"./.venvs/{env_name}")
            cmd = f"source {venv_path}/bin/activate && pip install {packages}"
    else:
        return json.dumps({
            "success": False,
            "error": "Either requirements or requirements_file must be provided",
        })
    
    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=600,
        )
        
        if result.returncode == 0:
            return json.dumps({
                "success": True,
                "env_name": env_name,
                "installed": requirements or requirements_file,
            })
        else:
            return json.dumps({
                "success": False,
                "error": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
            })
    except subprocess.TimeoutExpired:
        return json.dumps({
            "success": False,
            "error": "Installation timed out",
        })


async def setup_docker(
    base_image: str = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    requirements_file: Optional[str] = None,
    output_path: str = "Dockerfile",
) -> str:
    req_file = requirements_file or "requirements.txt"
    
    dockerfile_content = DOCKERFILE_TEMPLATE.format(
        base_image=base_image,
        requirements_file=req_file,
    )
    
    output = Path(output_path)
    output.write_text(dockerfile_content)
    
    compose_content = """version: '3.8'

services:
  experiment:
    build: .
    volumes:
      - .:/workspace
      - ./data:/workspace/data
      - ./outputs:/workspace/outputs
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    shm_size: '16gb'
    command: bash
"""
    
    compose_path = output.parent / "docker-compose.yml"
    compose_path.write_text(compose_content)
    
    return json.dumps({
        "success": True,
        "dockerfile": str(output),
        "docker_compose": str(compose_path),
        "base_image": base_image,
        "build_cmd": "docker build -t research-exp .",
        "run_cmd": "docker-compose up -d",
    })


async def check_gpu_availability() -> str:
    nvidia_smi = shutil.which("nvidia-smi")
    
    if not nvidia_smi:
        return json.dumps({
            "gpu_available": False,
            "error": "nvidia-smi not found",
            "recommendation": "Use CPU or check CUDA installation",
        })
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        gpus.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total_mb": int(parts[2]),
                            "memory_free_mb": int(parts[3]),
                            "utilization_percent": int(parts[4]),
                        })
            
            return json.dumps({
                "gpu_available": True,
                "gpu_count": len(gpus),
                "gpus": gpus,
                "recommendation": f"Use GPU {gpus[0]['index']}" if gpus else None,
            })
        else:
            return json.dumps({
                "gpu_available": False,
                "error": result.stderr,
            })
    except subprocess.TimeoutExpired:
        return json.dumps({
            "gpu_available": False,
            "error": "nvidia-smi timed out",
        })


async def clone_baseline_repos(
    paper_ids: list[str],
    target_dir: str = "./baselines",
) -> str:
    from src.tools.aggregation import clone_paper_code
    
    results = []
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    for paper_id in paper_ids:
        result = await clone_paper_code(paper_id, target_dir)
        result_data = json.loads(result)
        results.append({
            "paper_id": paper_id,
            **result_data,
        })
    
    successful = sum(1 for r in results if r.get("success", False))
    
    return json.dumps({
        "total": len(paper_ids),
        "successful": successful,
        "target_dir": target_dir,
        "results": results,
    }, indent=2)


async def setup_datasets(
    datasets: list[str],
    data_dir: str = "./data",
) -> str:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    HF_DATASETS = {
        "gsm8k": "gsm8k",
        "math": "competition_math",
        "mmlu": "cais/mmlu",
        "hellaswag": "hellaswag",
        "squad": "squad",
        "natural_questions": "natural_questions",
        "triviaqa": "trivia_qa",
    }
    
    TORCHVISION_DATASETS = ["mnist", "cifar10", "cifar100", "imagenet", "coco"]
    
    results = []
    
    for dataset in datasets:
        dataset_lower = dataset.lower()
        
        if dataset_lower in HF_DATASETS or "/" in dataset:
            hf_name = HF_DATASETS.get(dataset_lower, dataset)
            
            script = f"""
from datasets import load_dataset
ds = load_dataset("{hf_name}", cache_dir="{data_path}")
print(f"Downloaded: {hf_name}")
"""
            results.append({
                "dataset": dataset,
                "type": "huggingface",
                "hf_name": hf_name,
                "download_script": script.strip(),
                "status": "script_generated",
            })
        
        elif dataset_lower in TORCHVISION_DATASETS:
            script = f"""
import torchvision
dataset = torchvision.datasets.{dataset.upper()}(root="{data_path}", download=True)
print(f"Downloaded: {dataset}")
"""
            results.append({
                "dataset": dataset,
                "type": "torchvision",
                "download_script": script.strip(),
                "status": "script_generated",
            })
        
        else:
            results.append({
                "dataset": dataset,
                "type": "unknown",
                "status": "manual_download_required",
                "suggestion": f"Search for {dataset} on HuggingFace Datasets or Papers with Code",
            })
    
    download_all_script = data_path / "download_datasets.py"
    script_content = "#!/usr/bin/env python3\n\n"
    script_content += "from datasets import load_dataset\n"
    script_content += "import torchvision\n\n"
    
    for r in results:
        if r.get("download_script"):
            script_content += f"# {r['dataset']}\n"
            script_content += r["download_script"] + "\n\n"
    
    download_all_script.write_text(script_content)
    
    return json.dumps({
        "data_dir": str(data_path),
        "datasets": results,
        "download_script": str(download_all_script),
        "note": "Run 'python download_datasets.py' to download all datasets",
    }, indent=2)
