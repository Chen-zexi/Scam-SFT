import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import torch
import json
from pathlib import Path
from typing import List, Dict, Tuple
import subprocess
import sys

dtype = torch.bfloat16
n_gpus = torch.cuda.device_count()
GPU_BUFFERS = tuple([torch.empty(2*256*2048, dtype=dtype, device=f"cuda:{i}") for i in range(n_gpus)])

sample_data_cache = None


def display_banner():
    print("=" * 70)
    print("SCAM DETECTION MODEL - TRAINING & INFERENCE")
    print("=" * 70)


def get_library_version(library_name: str) -> str:
    try:
        if library_name == "torch":
            import torch
            return torch.__version__
        elif library_name == "unsloth":
            import unsloth
            return unsloth.__version__
        elif library_name == "triton":
            import triton
            return triton.__version__
        elif library_name == "bitsandbytes":
            import bitsandbytes
            return bitsandbytes.__version__
        elif library_name == "transformers":
            import transformers
            return transformers.__version__
        elif library_name == "accelerate":
            import accelerate
            return accelerate.__version__
        elif library_name == "peft":
            import peft
            return peft.__version__
        elif library_name == "datasets":
            import datasets
            return datasets.__version__
        elif library_name == "trl":
            import trl
            return trl.__version__
        elif library_name == "xformers":
            import xformers
            return xformers.__version__
        elif library_name == "flash_attn":
            import flash_attn
            return flash_attn.__version__
        else:
            return "Unknown library"
    except ImportError:
        return "Not installed"
    except AttributeError:
        return "Version not available"


def get_cuda_version() -> str:
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'release' in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'release' in part.lower() and i + 1 < len(parts):
                            return parts[i + 1].rstrip(',')
        return "CUDA not found"
    except FileNotFoundError:
        return "nvcc not found"
    except Exception:
        return "Unable to determine"


def check_environment():
    print("\n" + "=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    
    print("\nSYSTEM INFORMATION")
    print("-" * 30)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    
    print("\nGPU & CUDA INFORMATION")
    print("-" * 30)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version (nvcc): {get_cuda_version()}")
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            memory_gb = gpu_props.total_memory / (1024**3)
            
            print(f"\nGPU {i}: {gpu_props.name}")
            print(f"  • Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            print(f"  • Total Memory: {memory_gb:.1f} GB")
            print(f"  • Multiprocessors: {gpu_props.multi_processor_count}")
            
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"  • Memory Allocated: {memory_allocated:.2f} GB")
                print(f"  • Memory Reserved: {memory_reserved:.2f} GB")
    else:
        print("No CUDA GPUs available")
    
    print("\nLIBRARY VERSIONS")
    print("-" * 30)
    
    libraries = [
        "torch",
        "unsloth", 
        "transformers",
        "bitsandbytes",
        "triton",
        "accelerate",
        "peft",
        "datasets",
        "trl",
        "xformers",
        "flash_attn"
    ]
    
    for lib in libraries:
        version = get_library_version(lib)
        print(f"{lib:<15}: {version}")
    
    print("\nATTENTION MECHANISMS")
    print("-" * 30)
    
    flash_attn_version = get_library_version("flash_attn")
    flash_attn_available = flash_attn_version != "Not installed"
    print(f"Flash Attention: {'Available' if flash_attn_available else 'Not Available'}")
    if flash_attn_available:
        print(f"  Version: {flash_attn_version}")
        
        try:
            import flash_attn
            print(f"  CUDA Compatible: Yes")
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"  CUDA Compatible: No - {str(e)}")
            else:
                print(f"  CUDA Compatible: Unknown")
    
    xformers_version = get_library_version("xformers")
    xformers_available = xformers_version != "Not installed"
    print(f"xformers: {'Available' if xformers_available else 'Not Available'}")
    if xformers_available:
        print(f"  Version: {xformers_version}")
        
        try:
            import xformers.ops
            print(f"  Memory Efficient Attention: Available")
        except ImportError:
            print(f"  Memory Efficient Attention: Not Available")
    
    try:
        import torch.nn.functional as F
        if hasattr(F, 'scaled_dot_product_attention'):
            print(f"PyTorch Native SDPA: Available (PyTorch >= 2.0)")
        else:
            print(f"PyTorch Native SDPA: Not Available")
    except:
        print(f"PyTorch Native SDPA: Unknown")

    print("\nPYTORCH CONFIGURATION")
    print("-" * 30)
    print(f"PyTorch Built with CUDA: {torch.version.cuda is not None}")
    print(f"cuDNN Available: {torch.backends.cudnn.is_available()}")
    print(f"cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    
    print(f"MPS Available (Apple): {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'N/A'}")
    print(f"XPU Available (Intel): {torch.xpu.is_available() if hasattr(torch, 'xpu') else 'N/A'}")
    
    print("\n" + "=" * 60)
    input("Press Enter to continue...")


def get_saved_models() -> List[Tuple[str, str, str]]:
    output_dir = Path("output")
    models = []
    
    if not output_dir.exists():
        return models
    
    for timestamp_dir in output_dir.iterdir():
        if timestamp_dir.is_dir():
            for model_dir in timestamp_dir.iterdir():
                if model_dir.is_dir() and model_dir.name.startswith("lora_"):
                    if (model_dir / "adapter_config.json").exists():
                        training_config_path = timestamp_dir / "training_config.json"
                        training_summary_path = timestamp_dir / "training_summary.json"
                        
                        info = "No training info available"
                        if training_summary_path.exists():
                            try:
                                with open(training_summary_path, 'r') as f:
                                    summary = json.load(f)
                                    training_stats = summary.get('training_stats', 'Unknown')
                                    steps, loss = training_stats[0], training_stats[1]
                                    info = f"Steps: {steps}, Final Loss: {loss}"
                            except:
                                pass
                        
                        models.append((str(model_dir), timestamp_dir.name, info))
    
    return sorted(models, key=lambda x: x[1], reverse=True)


def display_available_models(models: List[Tuple[str, str, str]]) -> None:
    if not models:
        print("No fine-tuned models found in the output directory.")
        return
    
    print("\nAvailable Fine-tuned Models:")
    print("-" * 50)
    for i, (model_path, timestamp, info) in enumerate(models, 1):
        model_name = Path(model_path).name
        print(f"{i}. {model_name}")
        print(f"   Timestamp: {timestamp}")
        print(f"   Info: {info}")
        print(f"   Path: {model_path}")
        print()


def select_model_from_list(models: List[Tuple[str, str, str]]) -> str:
    if not models:
        print("No models available to select from.")
        return None
    
    display_available_models(models)
    
    while True:
        try:
            choice = input(f"Select a model (1-{len(models)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(models):
                return models[choice_idx][0]  # Return model path
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")


def run_sft_training():
    print("\n" + "=" * 50)
    print("STARTING SUPERVISED FINE-TUNING (SFT)")
    print("=" * 50)
    
    try:
        from src.train_pipeline import FineTuningPipeline, create_default_configs
        
        configs = create_default_configs()
        model_config, lora_config, training_config, data_config, inference_config, save_config = configs
    except Exception as e:
        print(f"Failed to import training dependencies: {e}")
        return False
    
    global sample_data_cache
    if inference_config.run_sample_comparison:
        try:
            print("\nSampling data for before/after training comparison...")
            from src.data_processor import DataProcessor
            from transformers import AutoTokenizer
            temp_tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
            data_processor = DataProcessor(data_config, temp_tokenizer)
            
            scam_data, legitimate_data = data_processor.sample_comparison_data(
                random_state=inference_config.sample_random_state
            )
            sample_data_cache = {'scam': scam_data, 'legitimate': legitimate_data}
            
            print("Sample data cached for comparison.")
        except Exception as e:
            print(f"Failed to sample comparison data: {e}")
            print("Training will continue without sample comparison.")
    
    pipeline = FineTuningPipeline(
        model_config=model_config,
        lora_config=lora_config,
        training_config=training_config,
        data_config=data_config,
        inference_config=inference_config,
        save_config=save_config
    )
    
    try:
        results = pipeline.run_full_pipeline(resume_from_checkpoint=False)
        print("\n" + "=" * 50)
        print("SFT TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        return True
    except Exception as e:
        print(f"\nSFT Training failed with error: {e}")
        return False


def run_base_model_inference():
    print("\n" + "=" * 50)
    print("RUNNING INFERENCE WITH BASE MODEL")
    print("=" * 50)
    
    try:
        from src.train_pipeline import create_default_configs
        from src.model_loader import ModelLoader
        from src.inference import ModelInference
        
        configs = create_default_configs()
        model_config, lora_config, training_config, data_config, inference_config, save_config = configs
    except Exception as e:
        print(f"Failed to import inference dependencies: {e}")
        return False
    
    try:
        print(f"Loading base model: {model_config.model_name}")
        
        model_loader = ModelLoader(model_config, lora_config)
        model, tokenizer = model_loader.load_base_model()
        
        inference = ModelInference(model, tokenizer, inference_config)
        
        if inference_config.run_sample_comparison:
            from src.data_processor import DataProcessor
            data_processor = DataProcessor(data_config, tokenizer)
            
            try:
                print("\nSampling data for comparison...")
                scam_data, legitimate_data = data_processor.sample_comparison_data(
                    random_state=inference_config.sample_random_state
                )
                
                inference.run_sample_comparison(scam_data, legitimate_data, "BASE MODEL")
                
                global sample_data_cache
                sample_data_cache = {'scam': scam_data, 'legitimate': legitimate_data}
                
            except Exception as e:
                print(f"Sample comparison failed: {e}")
                print("Continuing with interactive mode...")
        
        print("\nModel loaded successfully! Starting interactive inference...")
        print("You can now test the base model's capabilities.")
        
        inference.interactive_classification()
        
    except Exception as e:
        print(f"Base model inference failed with error: {e}")
        return False
    
    return True


def run_finetuned_model_inference():
    """Load and run inference with a fine-tuned model"""
    print("\n" + "=" * 50)
    print("LOADING FINE-TUNED MODEL FOR INFERENCE")
    print("=" * 50)
    
    try:
        from src.train_pipeline import create_default_configs
        from src.model_loader import ModelLoader
        from src.inference import ModelInference
    except Exception as e:
        print(f"Failed to import inference dependencies: {e}")
        return False
    
    models = get_saved_models()
    
    if not models:
        print("No fine-tuned models found in the output directory.")
        print("Please run SFT training first to create a fine-tuned model.")
        return False
    
    selected_model_path = select_model_from_list(models)
    
    if not selected_model_path:
        print("No model selected. Returning to main menu.")
        return False
    
    try:
        print(f"\nLoading fine-tuned model from: {selected_model_path}")
        
        configs = create_default_configs()
        model_config, lora_config, training_config, data_config, inference_config, save_config = configs
        
        model_loader = ModelLoader(model_config, lora_config)
        model, tokenizer = model_loader.load_pretrained_lora(selected_model_path)
        
        inference = ModelInference(model, tokenizer, inference_config)
        
        if inference_config.run_sample_comparison:
            try:
                global sample_data_cache
                if 'sample_data_cache' in globals() and sample_data_cache:
                    print("\nUsing cached sample data for comparison...")
                    scam_data = sample_data_cache['scam']
                    legitimate_data = sample_data_cache['legitimate']
                else:
                    print("\nNo cached sample data found. Sampling new data...")
                    from src.data_processor import DataProcessor
                    data_processor = DataProcessor(data_config, tokenizer)
                    scam_data, legitimate_data = data_processor.sample_comparison_data(
                        random_state=inference_config.sample_random_state
                    )
                
                model_name = f"FINE-TUNED MODEL ({Path(selected_model_path).name})"
                
                inference.run_sample_comparison(scam_data, legitimate_data, model_name)
                
            except Exception as e:
                print(f"Sample comparison failed: {e}")
                print("Continuing with interactive mode...")
        
        print("\nFine-tuned model loaded successfully! Starting interactive inference...")
        print("You can now test the fine-tuned model's scam detection capabilities.")
        
        inference.interactive_classification()
        
    except Exception as e:
        print(f"Fine-tuned model inference failed with error: {e}")
        return False
    
    return True


def run_model_conversion():
    print("\n" + "=" * 50)
    print("MODEL FORMAT CONVERSION")
    print("=" * 50)
    
    try:
        from src.train_pipeline import create_default_configs
        from src.model_converter import ModelConverter
        
        configs = create_default_configs()
        model_config, lora_config, training_config, data_config, inference_config, save_config = configs
    except Exception as e:
        print(f"Failed to import conversion dependencies: {e}")
        return False
    
    try:
        converter = ModelConverter(model_config, lora_config, save_config)
        
        success = converter.run_conversion_interface()
        
        if success:
            print("\n" + "=" * 50)
            print("MODEL CONVERSION COMPLETED SUCCESSFULLY!")
            print("=" * 50)
        
        return success
        
    except Exception as e:
        print(f"Model conversion failed with error: {e}")
        return False


def run_vllm_server():
    print("\n" + "=" * 50)
    print("vLLM SERVER - OpenAI COMPATIBLE API")
    print("=" * 50)
    
    try:
        from src.config import ModelConfig, LoRAConfig
        from src.vllm_server import VLLMServer
        
        model_config = ModelConfig()
        lora_config = LoRAConfig()
    except Exception as e:
        print(f"Failed to import vLLM server dependencies: {e}")
        return False
    
    try:
        server = VLLMServer(model_config, lora_config)
        
        success = server.run_server_interface()
        
        if success:
            print("\n" + "=" * 50)
            print("vLLM SERVER SESSION COMPLETED")
            print("=" * 50)
        
        return success
        
    except Exception as e:
        print(f"vLLM server failed with error: {e}")
        return False


def display_menu():
    print("\nPlease select an option:")
    print("1. Run Supervised Fine-Tuning (SFT)")
    print("2. Inference with Base Model")
    print("3. Inference with Fine-tuned Model")
    print("4. Convert Model to Different Formats")
    print("5. Serve Model with vLLM (OpenAI Compatible)")
    print("6. Check Environment")
    print("7. Exit")


def main():
    display_banner()
    
    while True:
        display_menu()
        
        try:
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == "1":
                run_sft_training()
            
            elif choice == "2":
                run_base_model_inference()
            
            elif choice == "3":
                run_finetuned_model_inference()
            
            elif choice == "4":
                run_model_conversion()
            
            elif choice == "5":
                run_vllm_server()
            
            elif choice == "6":
                check_environment()
            
            elif choice == "7":
                print("\nGoodbye!")
                break
            
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, 5, 6, or 7.")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()