import os
import json
import subprocess
import sys
from typing import Optional, List, Dict, Any
from pathlib import Path
from .config import ModelConfig, LoRAConfig, SaveConfig
from .model_loader import ModelLoader
from .model_saver import ModelSaver

class ModelConverter:
    
    def __init__(self, model_config: ModelConfig, lora_config: LoRAConfig, save_config: SaveConfig):
        self.model_config = model_config
        self.lora_config = lora_config
        self.save_config = save_config
        self.model_loader = ModelLoader(model_config, lora_config)
    
    def get_available_models(self) -> List[tuple]:
        output_dir = Path("output")
        models = []
        
        if not output_dir.exists():
            return models
        
        for timestamp_dir in output_dir.iterdir():
            if timestamp_dir.is_dir():
                for model_dir in timestamp_dir.iterdir():
                    if model_dir.is_dir() and model_dir.name.startswith("lora_"):
                        if (model_dir / "adapter_config.json").exists():
                            training_summary_path = timestamp_dir / "training_summary.json"
                            info = "No training info"
                            if training_summary_path.exists():
                                try:
                                    with open(training_summary_path, 'r') as f:
                                        summary = json.load(f)
                                        training_stats = summary.get('training_stats', {})
                                        if isinstance(training_stats, list) and len(training_stats) >= 2:
                                            steps, loss = training_stats[0], training_stats[1]
                                            info = f"Steps: {steps}, Loss: {loss:.4f}"
                                except:
                                    pass
                            models.append((str(model_dir), timestamp_dir.name, info))
        
        return sorted(models, key=lambda x: x[1], reverse=True)
    
    def display_conversion_options(self) -> None:
        print("\nAvailable conversion formats:")
        print("1. Merged 16-bit (vllm compatible [Recommended - Most Reliable])")
        print("2. Merged 4-bit (vllm compatible [Warning: Known Unsloth bugs])")
        print("3. GGUF Q8_0 (llama.cpp compatible)")
        print("4. GGUF Q4_0 (llama.cpp compatible)")
        print("5. GGUF Q5_0 (llama.cpp compatible)")
        print("6. GGUF Q5_1 (llama.cpp compatible)")
        print("7. GGUF Q6_K (llama.cpp compatible)")
        print("8. Multiple GGUF formats")
        print("9. All formats")
        print("\nNote: Option 2 (4-bit) has known bugs in Unsloth and may fail.")
    
    def get_conversion_choice(self) -> tuple:
        while True:
            try:
                choice = input("\nSelect conversion format (1-9) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return None, None
                
                choice_num = int(choice)
                if 1 <= choice_num <= 9:
                    format_map = {
                        1: ("merged", "merged_16bit"),
                        2: ("merged", "merged_4bit_forced"),
                        3: ("gguf", "q8_0"),
                        4: ("gguf", "q4_0"),
                        5: ("gguf", "q5_0"),
                        6: ("gguf", "q5_1"),
                        7: ("gguf", "q6_k"),
                        8: ("gguf_multiple", ["q4_0", "q5_0", "q8_0"]),
                        9: ("all", None)
                    }
                    return format_map[choice_num]
                else:
                    print("Please enter a number between 1 and 9")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
    
    def convert_model(self, model_path: str, output_format: str, format_option: Any) -> bool:
        try:
            print(f"\nLoading model from: {model_path}")
            
            # Load the fine-tuned model
            model, tokenizer = self.model_loader.load_pretrained_lora(model_path)
            
            # Create output directory for conversions
            conversion_dir = os.path.join(os.path.dirname(model_path), "conversions")
            os.makedirs(conversion_dir, exist_ok=True)
            
            # Create model saver for conversion
            model_saver = ModelSaver(model, tokenizer, self.save_config)
            
            print(f"Converting model to {output_format} format...")
            
            if output_format == "merged":
                merged_dir = os.path.join(conversion_dir, f"merged_{format_option}")
                model_saver.save_merged_model(merged_dir, save_method=format_option)
                print(f" Merged model saved to: {merged_dir}")
            
            elif output_format == "gguf":
                gguf_dir = os.path.join(conversion_dir, f"gguf_{format_option}")
                model_saver.save_gguf(gguf_dir, quantization_method=format_option)
                print(f" GGUF model saved to: {gguf_dir}")
            
            elif output_format == "gguf_multiple":
                gguf_dir = os.path.join(conversion_dir, "gguf_multiple")
                model_saver.save_multiple_gguf(gguf_dir, quantization_methods=format_option)
                print(f" Multiple GGUF models saved to: {gguf_dir}")
            
            elif output_format == "all":
                # Save all formats
                print("Converting to all formats...")
                
                # Merged 16-bit
                merged_16_dir = os.path.join(conversion_dir, "merged_16bit")
                model_saver.save_merged_model(merged_16_dir, save_method="merged_16bit")
                print(f" Merged 16-bit saved to: {merged_16_dir}")
                
                # Merged 4-bit
                merged_4_dir = os.path.join(conversion_dir, "merged_4bit")
                model_saver.save_merged_model(merged_4_dir, save_method="merged_4bit_forced")
                print(f" Merged 4-bit saved to: {merged_4_dir}")
                
                # Multiple GGUF formats
                gguf_dir = os.path.join(conversion_dir, "gguf_all")
                model_saver.save_multiple_gguf(gguf_dir, quantization_methods=["q4_0", "q5_0", "q8_0"])
                print(f" Multiple GGUF models saved to: {gguf_dir}")
            
            print(f"\n Model conversion completed successfully!")
            return True
            
        except Exception as e:
            print(f" Model conversion failed: {e}")
            return False
    
    def run_conversion_interface(self) -> bool:
        print("\n" + "=" * 60)
        print("MODEL FORMAT CONVERSION")
        print("=" * 60)
        
        # Get available models
        models = self.get_available_models()
        
        if not models:
            print("No fine-tuned models found for conversion.")
            print("Please train a model first using the SFT option.")
            return False
        
        # Display available models
        print("\nAvailable models for conversion:")
        print("-" * 50)
        for i, (model_path, timestamp, info) in enumerate(models, 1):
            model_name = Path(model_path).name
            print(f"{i}. {model_name}")
            print(f"   Timestamp: {timestamp}")
            print(f"   Info: {info}")
            print()
        
        # Let user select model
        while True:
            try:
                choice = input(f"Select model to convert (1-{len(models)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return False
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(models):
                    selected_model = models[choice_idx][0]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(models)}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
        
        # Display conversion options
        self.display_conversion_options()
        
        # Get conversion choice
        output_format, format_option = self.get_conversion_choice()
        if output_format is None:
            return False
        
        # Perform conversion
        success = self.convert_model(selected_model, output_format, format_option)
        
        if success:
            print("\n" + "=" * 60)
            print("MODEL CONVERSION COMPLETED!")
            print("=" * 60)
        
        return success