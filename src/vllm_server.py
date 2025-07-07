import os
import json
import subprocess
import sys
import signal
import time
import requests
import atexit
import threading
from typing import Optional, List, Dict, Any
from pathlib import Path
from .config import ModelConfig, LoRAConfig

# Global torch import for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class VLLMServer:
    
    def __init__(self, model_config: ModelConfig, lora_config: LoRAConfig):
        self.model_config = model_config
        self.lora_config = lora_config
        self.server_process = None
        self.server_url = None
        self.server_port = None
        self.shutdown_event = threading.Event()
        self.cleanup_registered = False
    
    def check_vllm_installation(self) -> bool:
        try:
            import vllm
            return True
        except ImportError:
            return False
    
    def install_vllm(self) -> bool:
        print("vLLM not found. Installing vLLM...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm"])
            print("vLLM installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install vLLM: {e}")
            return False
    
    def register_cleanup_handlers(self) -> None:
        if not self.cleanup_registered:
            atexit.register(self.force_stop_server)
            
            def signal_handler(signum, frame):
                print(f"\nReceived signal {signum}, shutting down server...")
                self.graceful_stop_server()
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            if sys.platform == "win32":
                signal.signal(signal.SIGBREAK, signal_handler)
            
            self.cleanup_registered = True
            print("Cleanup handlers registered")
    
    def get_available_models(self) -> List[tuple]:
        """Get list of available local models."""
        models = []
        output_dir = Path("output")
        
        if not output_dir.exists():
            return models
        
        for timestamp_dir in output_dir.iterdir():
            if timestamp_dir.is_dir():
                conversions_dir = timestamp_dir / "conversions"
                if conversions_dir.exists():
                    for conv_dir in conversions_dir.iterdir():
                        if conv_dir.is_dir() and conv_dir.name.startswith("merged_"):
                            if (conv_dir / "config.json").exists():
                                model_type = conv_dir.name.replace("merged_", "")
                                models.append((str(conv_dir), f"Converted-{model_type}", timestamp_dir.name))
        
        for timestamp_dir in output_dir.iterdir():
            if timestamp_dir.is_dir():
                for model_dir in timestamp_dir.iterdir():
                    if model_dir.is_dir() and model_dir.name.startswith("lora_"):
                        if (model_dir / "adapter_config.json").exists():
                            models.append((str(model_dir), "LoRA-Adapter", timestamp_dir.name))
        
        return sorted(models, key=lambda x: x[2], reverse=True)
    
    def get_available_lora_adapters(self) -> List[Dict[str, str]]:
        """Get list of available LoRA adapters with metadata."""
        adapters = []
        output_dir = Path("output")
        
        if not output_dir.exists():
            return adapters
        
        for timestamp_dir in output_dir.iterdir():
            if timestamp_dir.is_dir():
                for model_dir in timestamp_dir.iterdir():
                    if model_dir.is_dir() and model_dir.name.startswith("lora_"):
                        adapter_config_path = model_dir / "adapter_config.json"
                        if adapter_config_path.exists():
                            try:
                                base_model = self.get_base_model_from_adapter(str(adapter_config_path))
                                adapters.append({
                                    "path": str(model_dir),
                                    "name": model_dir.name,
                                    "base_model": base_model,
                                    "timestamp": timestamp_dir.name,
                                    "adapter_config": str(adapter_config_path)
                                })
                            except Exception as e:
                                print(f"Warning: Could not read adapter config from {adapter_config_path}: {e}")
        
        return sorted(adapters, key=lambda x: x["timestamp"], reverse=True)
    
    def get_base_model_from_adapter(self, adapter_config_path: str) -> str:
        """Extract base model name from adapter config."""
        try:
            with open(adapter_config_path, 'r') as f:
                config = json.load(f)
                return config.get("base_model_name_or_path", "unknown")
        except Exception as e:
            print(f"Error reading adapter config: {e}")
            return "unknown"
    
    def get_compatible_adapters(self, base_model: str) -> List[Dict[str, str]]:
        """Get LoRA adapters compatible with the specified base model."""
        all_adapters = self.get_available_lora_adapters()
        compatible = []
        
        for adapter in all_adapters:
            if adapter["base_model"] == base_model:
                compatible.append(adapter)
        
        return compatible
    
    def is_quantized_model(self, model_name: str) -> bool:
        """Check if a model is quantized (BitsAndBytes)."""
        quantized_indicators = [
            "bnb-4bit", "bnb-8bit", "4bit", "8bit", 
            "unsloth", "gptq", "awq", "eetq"
        ]
        
        model_lower = model_name.lower()
        return any(indicator in model_lower for indicator in quantized_indicators)
    
    def validate_huggingface_model(self, model_name: str) -> bool:
        """Validate if a Hugging Face model exists and is accessible."""
        try:
            import requests
            
            # Check if model exists on Hugging Face Hub
            url = f"https://huggingface.co/api/models/{model_name}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                model_info = response.json()
                print(f"✓ Found model: {model_name}")
                print(f"  Downloads: {model_info.get('downloads', 'N/A')}")
                print(f"  Library: {model_info.get('library_name', 'N/A')}")
                if 'tags' in model_info:
                    tags = [tag for tag in model_info['tags'] if tag in ['text-generation', 'text2text-generation']]
                    if tags:
                        print(f"  Task: {', '.join(tags)}")
                return True
            else:
                print(f"✗ Model '{model_name}' not found on Hugging Face Hub")
                return False
                
        except Exception as e:
            print(f"⚠ Could not validate model '{model_name}': {e}")
            print("Proceeding anyway - vLLM will validate during startup")
            return True  # Allow user to proceed, vLLM will handle validation
    
    def get_gpu_count(self) -> int:
        """Detect number of available GPUs."""
        if not TORCH_AVAILABLE:
            return 0
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()
    
    def display_server_options(self) -> None:
        """Display server configuration options."""
        gpu_count = self.get_gpu_count()
        
        print("\nServer Configuration Options:")
        print(f"Detected GPUs: {gpu_count}")
        print("1. Simple setup (auto-detect GPUs, host 0.0.0.0:8000)")
        print("2. Advanced configuration (custom settings)")
        print("3. Load from vllm_config.json (only --model will be set automatically)")
        
        if gpu_count > 1:
            print(f"   Note: {gpu_count} GPUs detected - tensor parallel will be enabled automatically")
        elif gpu_count == 1:
            print("   Note: Single GPU detected")
        else:
            print("   Warning: No GPUs detected - CPU only mode")
    
    def get_simple_config(self) -> Dict[str, Any]:
        """Get simple auto-configured setup."""
        gpu_count = self.get_gpu_count()
        
        config = {
            "host": "0.0.0.0",
            "port": 8000,
            "tensor_parallel_size": max(1, gpu_count) if gpu_count > 1 else 1,
            "gpu_count": gpu_count,
            "max_model_len": None,
            "reasoning_parser": None,
            "trust_remote_code": True,
            "enable_prompt_tokens_details": False,
            "max_num_batched_tokens": None,
            "gpu_memory_utilization": None,
            "enable_chunked_prefill": False,
            "enable_lora": False,
            "lora_modules": {},
            "max_loras": 1,
            "max_lora_rank": 32,
            "lora_dtype": "auto"
        }
        
        print(f"\nSimple setup configured:")
        print(f"  Host: {config['host']}")
        print(f"  Port: {config['port']}")
        print(f"  GPUs: {gpu_count}")
        if gpu_count > 1:
            print(f"  Tensor Parallel: {config['tensor_parallel_size']}")
        
        return config
    
    def get_advanced_config(self) -> Dict[str, Any]:
        """Get advanced configuration with user prompts."""
        gpu_count = self.get_gpu_count()
        
        config = {
            "host": "0.0.0.0",
            "port": 8000,
            "tensor_parallel_size": 1,
            "gpu_count": gpu_count,
            "max_model_len": None,
            "reasoning_parser": None,
            "trust_remote_code": True,
            "enable_prompt_tokens_details": False,
            "max_num_batched_tokens": None,
            "gpu_memory_utilization": None,
            "enable_chunked_prefill": False,
            "enable_lora": False,
            "lora_modules": {},
            "max_loras": 1,
            "max_lora_rank": 32,
            "lora_dtype": "auto"
        }
        
        print("\nAdvanced Configuration:")
        print("-" * 30)
        
        # Host configuration
        host_input = input("Host [0.0.0.0]: ").strip()
        if host_input:
            config["host"] = host_input
        
        # Port configuration
        port_input = input("Port [8000]: ").strip()
        if port_input:
            try:
                config["port"] = int(port_input)
            except ValueError:
                print("Invalid port, using default 8000")
        
        # Tensor parallel size
        tp_input = input(f"Tensor parallel size (detected {gpu_count} GPUs) [2]: ").strip()
        if tp_input:
            try:
                config["tensor_parallel_size"] = int(tp_input)
            except ValueError:
                config["tensor_parallel_size"] = 2
        else:
            config["tensor_parallel_size"] = 2
        
        # Max model length
        max_len_input = input("Max model length [10000]: ").strip()
        if max_len_input:
            try:
                config["max_model_len"] = int(max_len_input)
            except ValueError:
                config["max_model_len"] = 10000
        else:
            config["max_model_len"] = 10000
        
        # Reasoning parser
        reasoning_input = input("Reasoning parser [qwen3]: ").strip()
        if reasoning_input:
            config["reasoning_parser"] = reasoning_input
        elif reasoning_input in ['N', 'n', 'No', 'no', 'NO']:
            config["reasoning_parser"] = None
        else:
            config["reasoning_parser"] = "qwen3"
        
        # Enable prompt tokens details
        prompt_tokens = input("Enable prompt tokens details? (y/n) [n]: ").strip().lower()
        config["enable_prompt_tokens_details"] = prompt_tokens == 'y'
        
        # Max num batched tokens
        batched_tokens_input = input("Max num batched tokens [4096]: ").strip()
        if batched_tokens_input:
            try:
                config["max_num_batched_tokens"] = int(batched_tokens_input)
            except ValueError:
                config["max_num_batched_tokens"] = 4096
        else:
            config["max_num_batched_tokens"] = 4096
        
        # GPU memory utilization
        gpu_mem_input = input("GPU memory utilization (0.0-1.0) [0.90]: ").strip()
        if gpu_mem_input:
            try:
                config["gpu_memory_utilization"] = float(gpu_mem_input)
            except ValueError:
                config["gpu_memory_utilization"] = 0.90
        else:
            config["gpu_memory_utilization"] = 0.90
        
        # Enable chunked prefill
        chunked_prefill = input("Enable chunked prefill? (y/n) [y]: ").strip().lower()
        config["enable_chunked_prefill"] = chunked_prefill != 'n'
        
        # LoRA configuration (for non-LoRA servers, these will be ignored)
        print("\nLoRA Configuration (optional):")
        max_loras_input = input("Max LoRAs [1]: ").strip()
        if max_loras_input:
            try:
                config["max_loras"] = int(max_loras_input)
            except ValueError:
                config["max_loras"] = 1
        
        max_lora_rank_input = input("Max LoRA rank [32]: ").strip()
        if max_lora_rank_input:
            try:
                config["max_lora_rank"] = int(max_lora_rank_input)
            except ValueError:
                config["max_lora_rank"] = 32
        
        lora_dtype_input = input("LoRA dtype [auto]: ").strip()
        if lora_dtype_input:
            config["lora_dtype"] = lora_dtype_input
        
        return config
    
    def get_json_config(self, config_path: str = "vllm_config.json") -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            if not os.path.exists(config_path):
                print(f"Configuration file '{config_path}' not found.")
                print("Please create the file or use a different option.")
                return None
            
            with open(config_path, 'r') as f:
                json_config = json.load(f)
            
            print(f"Loaded configuration from '{config_path}':")
            print("-" * 30)
            for key, value in json_config.items():
                print(f"  {key}: {value}")
            print("-" * 30)
            
            return {"json_config": json_config, "config_path": config_path}
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {e}")
            return None
        except Exception as e:
            print(f"Error reading configuration file: {e}")
            return None
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration based on user choice."""
        print("\nConfiguring vLLM server...")
        self.display_server_options()
        
        while True:
            setup_choice = input("Select setup type (1: simple, 2: advanced, 3: json config) [1]: ").strip()
            if not setup_choice:
                setup_choice = "1"
            
            if setup_choice == "1":
                return self.get_simple_config()
            elif setup_choice == "2":
                return self.get_advanced_config()
            elif setup_choice == "3":
                config_path = input("JSON config file path [vllm_config.json]: ").strip()
                if not config_path:
                    config_path = "vllm_config.json"
                return self.get_json_config(config_path)
            else:
                print("Please enter 1, 2, or 3")
    
    def start_server_with_json_config(self, model_path: str, server_config: Dict[str, Any]) -> bool:
        """Start server using JSON configuration file."""
        try:
            json_config = server_config["json_config"]
            config_path = server_config["config_path"]
            
            cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_path
            ]
            
            # Process each argument from JSON config
            for key, value in json_config.items():
                if key == "model":  # Skip model as it's already set
                    continue
                    
                arg_name = f"--{key}"
                
                if isinstance(value, bool):
                    # For boolean flags (True = include flag, False = exclude flag)
                    if value:
                        cmd.append(arg_name)
                elif value is not None:
                    # For arguments with values
                    cmd.extend([arg_name, str(value)])
            
            # Set up CUDA environment based on tensor-parallel-size if specified
            env = os.environ.copy()
            tensor_parallel = json_config.get("tensor-parallel-size", 1)
            if isinstance(tensor_parallel, int) and tensor_parallel > 1:
                gpu_ids = ",".join(str(i) for i in range(tensor_parallel))
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids
            elif self.get_gpu_count() >= 1:
                env["CUDA_VISIBLE_DEVICES"] = "0"
            
            print(f"Starting vLLM server with JSON config from '{config_path}':")
            print(" ".join(cmd))
            print("\n" + "=" * 60)
            print("STARTING vLLM SERVER (JSON CONFIG)")
            print("=" * 60)
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env,
                preexec_fn=os.setsid
            )
            
            # Extract host and port from JSON config for server URL
            host = json_config.get("host", "127.0.0.1")
            port = json_config.get("port", 8000)
            self.server_url = f"http://{host}:{port}"
            self.server_port = port
            
            self.register_cleanup_handlers()
            
            print(f"Server starting at: {self.server_url}")
            print("Waiting for server to be ready...")
            
            if self.wait_for_server_ready(timeout=300):
                print("vLLM server is ready!")
                print(" Cleanup handlers active - server will shutdown gracefully on exit")
                return True
            else:
                print(" Server failed to start within timeout")
                self.stop_server()
                return False
                
        except Exception as e:
            print(f" Failed to start vLLM server with JSON config: {e}")
            return False
    
    def start_server(self, model_path: str, server_config: Dict[str, Any]) -> bool:
        
        try:
            # Check if this is a JSON config
            if "json_config" in server_config:
                return self.start_server_with_json_config(model_path, server_config)
            
            cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_path,
                "--host", server_config["host"],
                "--port", str(server_config["port"]),
                "--served-model-name", "scam-detection-model"
            ]
            
            # Tensor parallel configuration
            if server_config.get("tensor_parallel_size", 1) > 1:
                cmd.extend(["--tensor-parallel-size", str(server_config["tensor_parallel_size"])])
            
            # Max model length
            if server_config.get("max_model_len"):
                cmd.extend(["--max-model-len", str(server_config["max_model_len"])])
            
            # Reasoning parser
            if server_config.get("reasoning_parser"):
                cmd.extend(["--reasoning-parser", server_config["reasoning_parser"]])
            
            # Trust remote code (always enabled for compatibility)
            if server_config.get("trust_remote_code", True):
                cmd.append("--trust-remote-code")
            
            # Enable prompt tokens details
            if server_config.get("enable_prompt_tokens_details", False):
                cmd.append("--enable-prompt-tokens-details")
            
            # Max num batched tokens
            if server_config.get("max_num_batched_tokens"):
                cmd.extend(["--max-num-batched-tokens", str(server_config["max_num_batched_tokens"])])
            
            # GPU memory utilization
            if server_config.get("gpu_memory_utilization"):
                cmd.extend(["--gpu-memory-utilization", str(server_config["gpu_memory_utilization"])])
            
            # Enable chunked prefill
            if server_config.get("enable_chunked_prefill", False):
                cmd.append("--enable-chunked-prefill")
            
            # LoRA configuration
            if server_config.get("enable_lora", False):
                cmd.append("--enable-lora")
                
                # Add LoRA modules
                lora_modules = server_config.get("lora_modules", {})
                for name, path in lora_modules.items():
                    cmd.extend(["--lora-modules", f"{name}={path}"])
                
                # Max LoRAs
                if server_config.get("max_loras"):
                    cmd.extend(["--max-loras", str(server_config["max_loras"])])
                
                # Max LoRA rank
                if server_config.get("max_lora_rank"):
                    cmd.extend(["--max-lora-rank", str(server_config["max_lora_rank"])])
                
                # LoRA dtype
                if server_config.get("lora_dtype") and server_config["lora_dtype"] != "auto":
                    cmd.extend(["--lora-dtype", server_config["lora_dtype"]])
            
            # Disable log requests for cleaner output
            cmd.append("--disable-log-requests")
            
            # Set up CUDA environment
            env = os.environ.copy()
            gpu_count = server_config.get("gpu_count", self.get_gpu_count())
            if gpu_count > 1 and server_config.get("tensor_parallel_size", 1) > 1:
                # Set CUDA_VISIBLE_DEVICES for multi-GPU setup
                gpu_ids = ",".join(str(i) for i in range(server_config.get("tensor_parallel_size", 2)))
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids
            elif gpu_count >= 1:
                env["CUDA_VISIBLE_DEVICES"] = "0"
            
            print(f"Starting vLLM server with command:")
            print(" ".join(cmd))
            print("\n" + "=" * 60)
            print("STARTING vLLM SERVER")
            print("=" * 60)
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,  # Line buffered
                env=env,  # Pass environment variables
                preexec_fn=os.setsid  # Create process group for clean shutdown
            )
            
            self.server_url = f"http://{server_config['host']}:{server_config['port']}"
            self.server_port = server_config["port"]
            
            self.register_cleanup_handlers()
            
            print(f"Server starting at: {self.server_url}")
            print("Waiting for server to be ready...")
            
            if self.wait_for_server_ready(timeout=300):  # 5 minutes timeout
                print("vLLM server is ready!")
                print(" Cleanup handlers active - server will shutdown gracefully on exit")
                return True
            else:
                print(" Server failed to start within timeout")
                self.stop_server()
                return False
                
        except Exception as e:
            print(f" Failed to start vLLM server: {e}")
            return False
    
    def wait_for_server_ready(self, timeout: int = 300) -> bool:
        
        import threading
        import queue
        
        start_time = time.time()
        output_queue = queue.Queue()
        server_ready = False
        
        def read_output():
            
            for line in iter(self.server_process.stdout.readline, ''):
                output_queue.put(line.rstrip())
                if self.server_process.poll() is not None:
                    break
        
        output_thread = threading.Thread(target=read_output, daemon=True)
        output_thread.start()
        
        print("vLLM Server Output:")
        print("-" * 60)
        
        while time.time() - start_time < timeout:
            try:
                while True:
                    line = output_queue.get_nowait()
                    print(f"[vLLM] {line}")
                    
                    if "Uvicorn running on" in line or "Application startup complete" in line:
                        server_ready = True
            except queue.Empty:
                pass
            
            if server_ready:
                try:
                    response = requests.get(f"{self.server_url}/health", timeout=2)
                    if response.status_code == 200:
                        print("-" * 60)
                        print("✓ Server is ready and responding to health checks!")
                        return True
                except requests.exceptions.RequestException:
                    pass
            
            if self.server_process and self.server_process.poll() is not None:
                try:
                    while True:
                        line = output_queue.get_nowait()
                        print(f"[vLLM] {line}")
                except queue.Empty:
                    pass
                
                print("-" * 60)
                print(" Server process terminated unexpectedly")
                print(f"Exit code: {self.server_process.returncode}")
                return False
            
            time.sleep(2)
        
        print("-" * 60)
        print(" Server startup timeout reached")
        return False
    
    def graceful_stop_server(self) -> bool:
        
        if not self.server_process:
            return True
            
        try:
            print("Initiating graceful server shutdown...")
            
            self.shutdown_event.set()
            
            if hasattr(os, 'killpg') and hasattr(os, 'getpgid'):
                try:
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    self.server_process = None
                    return True
            else:
                self.server_process.terminate()
            
            print("⏳ Waiting for server to shut down gracefully...")
            for i in range(15):  # 15 seconds timeout
                try:
                    self.server_process.wait(timeout=1)
                    self.server_process = None
                    print("Server stopped gracefully")
                    return True
                except subprocess.TimeoutExpired:
                    print(f"⏳ Waiting... ({i+1}/15)")
                    continue
            
            print("Graceful shutdown timeout, forcing shutdown...")
            return self.force_stop_server()
                
        except Exception as e:
            print(f"Error during graceful shutdown: {e}")
            print("Attempting force shutdown...")
            return self.force_stop_server()
    
    def force_stop_server(self) -> bool:
        
        if not self.server_process:
            return True
            
        try:
            print("Force stopping server...")
            
            self.shutdown_event.set()
            
            if hasattr(os, 'killpg') and hasattr(os, 'getpgid'):
                try:
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                self.server_process.kill()
            
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Process still running after SIGKILL")
            
            self.server_process = None
            print("Server force stopped")
            return True
                
        except Exception as e:
            print(f"Error during force stop: {e}")
            self.server_process = None  # Reset anyway
            return False
    
    def stop_server(self) -> bool:
        
        return self.graceful_stop_server()
    
    def show_server_status(self) -> None:
        
        print("\n" + "=" * 40)
        print("SERVER STATUS")
        print("=" * 40)
        
        if self.server_process:
            if self.server_process.poll() is None:
                print("Status: Running")
                print(f"URL: {self.server_url}")
                print(f"Port: {self.server_port}")
                print(f"PID: {self.server_process.pid}")
                
                try:
                    response = requests.get(f"{self.server_url}/health", timeout=2)
                    if response.status_code == 200:
                        print("Health: OK")
                    else:
                        print(f"Health: HTTP {response.status_code}")
                except Exception:
                    print("Health: No response")
            else:
                print("Status: Stopped")
                print(f"Exit code: {self.server_process.returncode}")
        else:
            print("Status: Not started")
        
        print("=" * 40)
    
    def show_server_commands(self) -> None:
        
        print("\n" + "=" * 40)
        print("AVAILABLE COMMANDS")
        print("=" * 40)
        print("stop        - Graceful shutdown")
        print("force-stop  - Immediate shutdown")
        print("status      - Show server status")
        print("help        - Show this help")
        print("Ctrl+C     - Graceful shutdown")
        print("=" * 40)
    
    
    def run_server_interface(self) -> bool:
        
        if not self.check_vllm_installation():
            install_choice = input("vLLM is not installed. Install it now? (y/n): ").strip().lower()
            if install_choice == 'y':
                if not self.install_vllm():
                    return False
            else:
                print("vLLM is required to run the server.")
                return False
        
        # Get local models
        local_models = self.get_available_models()
        
        # Display model source options
        print("\n" + "=" * 60)
        print("MODEL SELECTION")
        print("=" * 60)
        print("Choose model source:")
        print("1. Use local trained/converted model")
        print("2. Use Hugging Face model (Only FP16 compatible)")
        print("3. Use LoRA adapter with base model")
        
        while True:
            source_choice = input("\nSelect model source (1: local, 2: huggingface, 3: lora) [1]: ").strip()
            if not source_choice:
                source_choice = "1"
            
            if source_choice == "1":
                # Local models
                if not local_models:
                    print("\nNo local models found for serving.")
                    print("Please convert a model first using the conversion option,")
                    print("or choose option 2 to use a Hugging Face model.")
                    continue
                
                print("\nAvailable local models:")
                print("-" * 50)
                for i, (model_path, model_type, timestamp) in enumerate(local_models, 1):
                    model_name = Path(model_path).name
                    print(f"{i}. {model_name} ({model_type})")
                    print(f"   Timestamp: {timestamp}")
                    if model_type.startswith("LoRA"):
                        print("   Note: vLLM works better with merged models")
                    print()
                
                while True:
                    try:
                        choice = input(f"Select local model (1-{len(local_models)}) or 'b' for back: ").strip()
                        if choice.lower() == 'b':
                            break
                        
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(local_models):
                            selected_model = local_models[choice_idx][0]
                            return self.start_server_with_config(selected_model)
                        else:
                            print(f"Please enter a number between 1 and {len(local_models)}")
                    except ValueError:
                        print("Please enter a valid number or 'b' to go back")
                
            elif source_choice == "2":
                # Hugging Face models
                print("\nHugging Face Model Selection")
                print("-" * 30)
                print("Popular models for text generation:")
                print("• Qwen/Qwen3-30B-A3B")
                print("• google/gemma-3n-E4B-it") 
                print("• baidu/ERNIE-4.5-21B-A3B-PT")
                print("• tencent/Hunyuan-A13B-Instruct")
                print("• Or any other text generation model from HuggingFace Hub")
                
                while True:
                    hf_model = input("\nEnter Hugging Face model name (or 'b' for back): ").strip()
                    if hf_model.lower() == 'b':
                        break
                    
                    if not hf_model:
                        print("Please enter a model name")
                        continue
                    
                    print(f"\nValidating model: {hf_model}")
                    if self.validate_huggingface_model(hf_model):
                        return self.start_server_with_config(hf_model)
                    else:
                        retry = input("Model validation failed. Try anyway? (y/n): ").strip().lower()
                        if retry == 'y':
                            return self.start_server_with_config(hf_model)
                            
            elif source_choice == "3":
                # LoRA adapter with base model
                return self.lora_adapter_selection()
                
            else:
                print("Please enter 1, 2, or 3")
        
        return False
    
    def lora_adapter_selection(self) -> bool:
        """Handle LoRA adapter selection process."""
        print("\n" + "=" * 60)
        print("LORA ADAPTER SELECTION")
        print("=" * 60)
        
        # Get available LoRA adapters
        lora_adapters = self.get_available_lora_adapters()
        
        if not lora_adapters:
            print("No LoRA adapters found.")
            print("Please train a model first using the SFT option.")
            return False
        
        print("Available LoRA adapters:")
        print("-" * 60)
        for i, adapter in enumerate(lora_adapters, 1):
            print(f"{i}. {adapter['name']}")
            print(f"   Base Model: {adapter['base_model']}")
            print(f"   Timestamp: {adapter['timestamp']}")
            print(f"   Path: {adapter['path']}")
            
            # Check if base model is quantized
            if self.is_quantized_model(adapter['base_model']):
                print(f"   ⚠️  Quantized model (single GPU only)")
            print()
        
        while True:
            try:
                choice = input(f"Select LoRA adapter (1-{len(lora_adapters)}) or 'b' for back: ").strip()
                if choice.lower() == 'b':
                    return False
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(lora_adapters):
                    selected_adapter = lora_adapters[choice_idx]
                    return self.start_lora_server(selected_adapter)
                else:
                    print(f"Please enter a number between 1 and {len(lora_adapters)}")
            except ValueError:
                print("Please enter a valid number or 'b' to go back")
    
    def start_lora_server(self, adapter_info: Dict[str, str]) -> bool:
        """Start server with LoRA adapter configuration."""
        print(f"\n" + "=" * 60)
        print("LORA SERVER CONFIGURATION")
        print("=" * 60)
        print(f"Selected adapter: {adapter_info['name']}")
        print(f"Base model: {adapter_info['base_model']}")
        print(f"Adapter path: {adapter_info['path']}")
        
        # Use the base model as the main model
        base_model = adapter_info['base_model']
        
        # Check if base model is quantized (BitsAndBytes)
        is_quantized = self.is_quantized_model(base_model)
        if is_quantized:
            print(f"\n⚠️  Detected quantized base model: {base_model}")
            print("   Quantized models don't support tensor parallelism in vLLM")
            print("   Forcing single GPU mode for compatibility")
        
        # Get server configuration
        server_config = self.get_server_config()
        
        if server_config is None:
            print("Failed to load configuration.")
            return False
        
        # Force single GPU for quantized models
        if is_quantized:
            server_config["tensor_parallel_size"] = 1
            server_config["gpu_count"] = 1
            print(f"   Overriding tensor parallel size to 1")
        
        # Add LoRA configuration to server config
        server_config["enable_lora"] = True
        server_config["lora_modules"] = {
            "scam_adapter": adapter_info['path']
        }
        
        if self.start_server(base_model, server_config):
            print("\n" + "=" * 60)
            print("LORA SERVER RUNNING")
            print("=" * 60)
            print(f"Base Model: {base_model}")
            print(f"LoRA Adapter: {adapter_info['name']}")
            print("Server control options:")
            print("   • Press Ctrl+C for graceful shutdown")
            print("   • Type 'stop' + Enter for graceful shutdown")
            print("   • Type 'force-stop' + Enter for immediate shutdown")
            print("   • Type 'status' + Enter for server status")
            print("   • Type 'help' + Enter for command list")
            print("=" * 60)
            
            # Start server monitoring
            return self.monitor_server()
        
        return False
    
    def start_server_with_config(self, selected_model: str) -> bool:
        """Start server with the selected model and get configuration."""
        
        server_config = self.get_server_config()
        
        # Handle case where JSON config loading failed
        if server_config is None:
            print("Failed to load configuration.")
            return False
        
        if self.start_server(selected_model, server_config):
            print("\n" + "=" * 60)
            print("SERVER RUNNING")
            print("=" * 60)
            print("Server control options:")
            print("   • Press Ctrl+C for graceful shutdown")
            print("   • Type 'stop' + Enter for graceful shutdown")
            print("   • Type 'force-stop' + Enter for immediate shutdown")
            print("   • Type 'status' + Enter for server status")
            print("   • Type 'help' + Enter for command list")
            print("=" * 60)
            
            # Start server monitoring
            return self.monitor_server()
            
        elif server_config.get("tensor_parallel_size", 1) > 1:
            print("\n" + "WARNING: " + "=" * 20)
            print("Tensor parallel setup failed. Trying single GPU fallback...")
            print("=" * 40)
            
            server_config["tensor_parallel_size"] = 1
            server_config["gpu_count"] = 1
            
            if self.start_server(selected_model, server_config):
                print("\n" + "=" * 60)
                print("SERVER RUNNING (Single GPU)")
                print("=" * 60)
                print("Press Ctrl+C to stop the server")
                
                # Start server monitoring
                return self.monitor_server()
        
        return False
    
    def monitor_server(self) -> bool:
        """Monitor running server and handle user commands."""
        try:
            import threading
            import queue
            
            output_queue = queue.Queue()
            command_queue = queue.Queue()
            
            def read_server_output():
                for line in iter(self.server_process.stdout.readline, ''):
                    output_queue.put(line.rstrip())
                    if self.server_process.poll() is not None:
                        break
            
            def read_user_commands():
                while not self.shutdown_event.is_set():
                    try:
                        if sys.platform != "win32":
                            import select
                            if select.select([sys.stdin], [], [], 0.1)[0]:
                                command = input().strip().lower()
                                command_queue.put(command)
                        else:
                            command = input().strip().lower()
                            command_queue.put(command)
                    except (EOFError, KeyboardInterrupt):
                        break
                    except Exception:
                        continue
            
            output_thread = threading.Thread(target=read_server_output, daemon=True)
            command_thread = threading.Thread(target=read_user_commands, daemon=True)
            output_thread.start()
            command_thread.start()
            
            print("\nMonitoring server output (Press Ctrl+C to stop):")
            print("-" * 60)
            
            while self.server_process and self.server_process.poll() is None and not self.shutdown_event.is_set():
                try:
                    while True:
                        line = output_queue.get_nowait()
                        print(f"[vLLM] {line}")
                except queue.Empty:
                    pass
                
                try:
                    while True:
                        command = command_queue.get_nowait()
                        if command == "stop":
                            print("\nUser requested graceful shutdown...")
                            self.graceful_stop_server()
                            return True
                        elif command == "force-stop":
                            print("\nUser requested force shutdown...")
                            self.force_stop_server()
                            return True
                        elif command == "status":
                            self.show_server_status()
                        elif command == "help":
                            self.show_server_commands()
                        else:
                            print(f"Unknown command: '{command}'. Type 'help' for available commands.")
                except queue.Empty:
                    pass
                
                time.sleep(0.1)
            
            try:
                while True:
                    line = output_queue.get_nowait()
                    print(f"[vLLM] {line}")
            except queue.Empty:
                pass
            
            if self.shutdown_event.is_set():
                print("-" * 60)
                print("Shutdown event detected")
                return True
            elif self.server_process and self.server_process.poll() != 0:
                print("-" * 60)
                print("Server process terminated unexpectedly")
                print(f"Exit code: {self.server_process.returncode}")
                return False
            else:
                print("-" * 60)
                print("Server stopped normally")
                return True
                
        except KeyboardInterrupt:
            print("\n" + "-" * 60)
            print("Keyboard interrupt received - initiating graceful shutdown...")
            self.graceful_stop_server()
            print("Server shutdown complete")
            return True