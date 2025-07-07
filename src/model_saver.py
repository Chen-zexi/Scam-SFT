import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from .config import SaveConfig

class ModelSaver:
    """Handles saving models in various formats"""
    
    def __init__(self, model, tokenizer, save_config: SaveConfig, model_name: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.save_config = save_config
        self.model_name = model_name or "unknown_model"
        self.output_dir = None
        self.training_config = None
        self.training_stats = None
    
    def setup_output_directory(self) -> str:
        """Setup and return the main output directory with timestamp"""
        if self.output_dir is None:
            self.output_dir = self.save_config.get_output_dir(self.model_name)
        
        # Create directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory created: {self.output_dir}")
        return self.output_dir
    
    def get_timestamp_directory(self) -> str:
        """Get the timestamp directory (parent of model directory)"""
        if self.output_dir is None:
            self.setup_output_directory()
        
        # Get the parent directory (timestamp directory)
        timestamp_dir = os.path.dirname(self.output_dir)
        return timestamp_dir
    
    def set_training_metadata(self, training_config: Dict[str, Any], training_stats: Optional[Dict[str, Any]] = None):
        """Set training configuration and statistics for saving"""
        self.training_config = training_config
        self.training_stats = training_stats
    
    def save_training_config(self, output_dir: str) -> str:
        """Save training configuration to JSON file"""
        if not self.training_config:
            print("No training configuration to save")
            return None
        
        config_path = os.path.join(output_dir, "training_config.json")
        
        # Convert any non-serializable objects to strings
        serializable_config = self._make_serializable(self.training_config)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)
        
        print(f"Training configuration saved to: {config_path}")
        return config_path
    
    def save_training_summary(self, output_dir: str) -> str:
        """Save training summary to JSON file"""
        if not self.training_stats:
            print("No training statistics to save")
            return None
        
        summary_path = os.path.join(output_dir, "training_summary.json")
        
        # Create comprehensive summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "output_directory": output_dir,
            "training_stats": self._make_serializable(self.training_stats),
            "save_config": self._make_serializable(self.save_config.__dict__)
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Training summary saved to: {summary_path}")
        return summary_path
    
    def _make_serializable(self, obj) -> Any:
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)
    
    def save_lora_adapters(self, output_dir: Optional[str] = None) -> str:
        """Save LoRA adapters locally"""
        if output_dir is None:
            output_dir = self.setup_output_directory()
        
        print(f"Saving LoRA adapters to: {output_dir}")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"LoRA adapters saved successfully to {output_dir}")
        return output_dir
    
    def push_lora_to_hub(self, repo_name: str, token: str) -> None:
        """Push LoRA adapters to Hugging Face Hub"""
        print(f"Pushing LoRA adapters to Hub: {repo_name}")
        
        self.model.push_to_hub(repo_name, token=token)
        self.tokenizer.push_to_hub(repo_name, token=token)
        
        print(f"LoRA adapters pushed successfully to {repo_name}")
    
    def save_merged_model(self, 
                         output_dir: str, 
                         save_method: str = "merged_16bit",
                         push_to_hub: bool = False,
                         hub_repo_name: Optional[str] = None,
                         token: Optional[str] = None) -> str:
        """Save merged model in specified format"""
        
        valid_methods = ["merged_16bit", "merged_4bit", "merged_4bit_forced", "lora"]
        if save_method not in valid_methods:
            raise ValueError(f"Invalid save_method. Must be one of: {valid_methods}")
        
        print(f"Saving merged model with method: {save_method}")
        
        # Handle the known Unsloth issues with 4-bit merging
        if save_method == "merged_4bit_forced":
            print("Warning: merged_4bit_forced has known bugs in Unsloth and may fail")
            print("This is due to internal Unsloth issues with 'base_model' variable scope")
            print("Recommending merged_16bit for best compatibility and reliability")
            
            # Try the forced method, but catch the known base_model error
            try:
                if push_to_hub:
                    if not hub_repo_name or not token:
                        raise ValueError("hub_repo_name and token are required for pushing to hub")
                    
                    print(f"Attempting to push merged model to Hub: {hub_repo_name}")
                    self.model.push_to_hub_merged(
                        hub_repo_name, 
                        self.tokenizer, 
                        save_method=save_method, 
                        token=token
                    )
                    print(f"Merged model pushed successfully to {hub_repo_name}")
                    return hub_repo_name
                else:
                    os.makedirs(output_dir, exist_ok=True)
                    print("Attempting 4-bit forced merge (may fail due to Unsloth bugs)...")
                    self.model.save_pretrained_merged(
                        output_dir, 
                        self.tokenizer, 
                        save_method=save_method
                    )
                    print(f"Merged model saved successfully to {output_dir}")
                    return output_dir
            except Exception as e:
                error_msg = str(e)
                if "base_model" in error_msg:
                    print(f"Known Unsloth bug encountered: {error_msg}")
                    print("This is a bug in Unsloth's 4-bit merging implementation")
                    print("Solution: Use option 1 (merged_16bit) which is more reliable")
                    raise Exception(f"Unsloth 4-bit merging bug: 'base_model' not defined. Use merged_16bit instead.")
                else:
                    print(f"4-bit merging failed with error: {error_msg}")
                    raise Exception(f"4-bit merging failed. Please use merged_16bit instead. Error: {error_msg}")
                    
        # Handle regular merged_4bit (will show warning but not forced)
        if save_method == "merged_4bit":
            print("Note: Unsloth will show a warning about accuracy loss for merged_4bit")
            print("The conversion will fail and suggest using merged_4bit_forced")
            print("However, merged_4bit_forced has known bugs. Consider merged_16bit instead.")
        
        if push_to_hub:
            if not hub_repo_name or not token:
                raise ValueError("hub_repo_name and token are required for pushing to hub")
            
            print(f"Pushing merged model to Hub: {hub_repo_name}")
            self.model.push_to_hub_merged(
                hub_repo_name, 
                self.tokenizer, 
                save_method=save_method, 
                token=token
            )
            print(f"Merged model pushed successfully to {hub_repo_name}")
            return hub_repo_name
        else:
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            self.model.save_pretrained_merged(
                output_dir, 
                self.tokenizer, 
                save_method=save_method
            )
            print(f"Merged model saved successfully to {output_dir}")
            return output_dir
    
    def save_gguf(self, 
                  output_dir: str,
                  quantization_method: str = "q8_0",
                  push_to_hub: bool = False,
                  hub_repo_name: Optional[str] = None,
                  token: Optional[str] = None) -> str:
        """Save model in GGUF format for llama.cpp"""
        
        print(f"Saving GGUF model with quantization: {quantization_method}")
        
        if push_to_hub:
            if not hub_repo_name or not token:
                raise ValueError("hub_repo_name and token are required for pushing to hub")
            
            print(f"Pushing GGUF model to Hub: {hub_repo_name}")
            self.model.push_to_hub_gguf(
                hub_repo_name,
                self.tokenizer,
                quantization_method=quantization_method,
                token=token
            )
            print(f"GGUF model pushed successfully to {hub_repo_name}")
            return hub_repo_name
        else:
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            self.model.save_pretrained_gguf(
                output_dir,
                self.tokenizer,
                quantization_method=quantization_method
            )
            print(f"GGUF model saved successfully to {output_dir}")
            return output_dir
    
    def save_multiple_gguf(self,
                          output_dir: str,
                          quantization_methods: List[str],
                          push_to_hub: bool = False,
                          hub_repo_name: Optional[str] = None,
                          token: Optional[str] = None) -> str:
        """Save model in multiple GGUF quantization formats"""
        
        print(f"Saving GGUF model with multiple quantizations: {quantization_methods}")
        
        if push_to_hub:
            if not hub_repo_name or not token:
                raise ValueError("hub_repo_name and token are required for pushing to hub")
            
            print(f"Pushing multiple GGUF models to Hub: {hub_repo_name}")
            self.model.push_to_hub_gguf(
                hub_repo_name,
                self.tokenizer,
                quantization_method=quantization_methods,
                token=token
            )
            print(f"Multiple GGUF models pushed successfully to {hub_repo_name}")
            return hub_repo_name
        else:
            # For local saving, save each quantization separately
            saved_paths = []
            for method in quantization_methods:
                method_output_dir = os.path.join(output_dir, method)
                os.makedirs(method_output_dir, exist_ok=True)
                
                self.model.save_pretrained_gguf(
                    method_output_dir,
                    self.tokenizer,
                    quantization_method=method
                )
                saved_paths.append(method_output_dir)
                print(f"GGUF model ({method}) saved to {method_output_dir}")
            
            return output_dir
    
    def save_all_formats(self) -> dict:
        """Save model in all configured formats"""
        results = {}
        
        # Setup main output directory with timestamp
        main_output_dir = self.setup_output_directory()
        timestamp_dir = self.get_timestamp_directory()
        
        # Always save LoRA adapters in the main directory
        results['lora'] = self.save_lora_adapters(main_output_dir)
        results['output_directory'] = main_output_dir
        results['timestamp_directory'] = timestamp_dir
        
        # Save merged model if configured
        if self.save_config.save_merged:
            merged_dir = os.path.join(main_output_dir, f"merged_{self.save_config.save_method}")
            results['merged'] = self.save_merged_model(
                merged_dir, 
                save_method=self.save_config.save_method,
                push_to_hub=self.save_config.push_to_hub,
                hub_repo_name=self.save_config.hub_model_name
            )
        
        # Save GGUF if configured
        if self.save_config.save_gguf:
            gguf_dir = os.path.join(main_output_dir, f"gguf_{self.save_config.gguf_quantization_method}")
            results['gguf'] = self.save_gguf(
                gguf_dir,
                quantization_method=self.save_config.gguf_quantization_method,
                push_to_hub=self.save_config.push_to_hub,
                hub_repo_name=self.save_config.hub_model_name
            )
        
        # Save metadata files in the timestamp directory (parent of model directory)
        if self.save_config.save_training_config and self.training_config:
            results['training_config'] = self.save_training_config(timestamp_dir)
        
        if self.save_config.save_training_summary and self.training_stats:
            results['training_summary'] = self.save_training_summary(timestamp_dir)
        
        return results
    
    def cleanup_temp_files(self) -> None:
        """Clean up any temporary files created during saving"""
        # This can be extended to clean up specific temporary files
        print("Cleaning up temporary files...")
        # Add cleanup logic if needed
        print("Cleanup completed") 