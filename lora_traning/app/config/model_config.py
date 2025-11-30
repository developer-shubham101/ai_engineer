# Model Configuration - Single Source of Truth
class ModelConfig:
    # Default model settings - UPDATE ONLY HERE
    DEFAULT_BASE_MODEL = "gpt2"  # Change this
    DEFAULT_OUTPUT_NAME = "gpt2-company-tuned"  # Change this

    
    # Training parameters
    DEFAULT_EPOCHS = 3
    DEFAULT_LEARNING_RATE = 2e-5
    DEFAULT_MAX_SAMPLES = 500
    
    @classmethod
    def get_model_config(cls, model_name: str = None):
        """Get model configuration by name or default"""
        return {
            "base_model": model_name or cls.DEFAULT_BASE_MODEL,
            "output_name": f"{(model_name or cls.DEFAULT_BASE_MODEL).replace('/', '-')}-company-tuned",
            "epochs": cls.DEFAULT_EPOCHS,
            "learning_rate": cls.DEFAULT_LEARNING_RATE,
            "max_samples": cls.DEFAULT_MAX_SAMPLES
        }