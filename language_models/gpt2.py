import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2Chat:
    """Model for generating responses using GPT-2."""
    def __init__(self, model_name='gpt2'):
        """Initialize the model and tokenizer.
        Args:
            model_name (str): Name of the GPT-2 model to use.
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    def generate_response(self, prompt, max_length=150):
        """Generate a text response based on the input prompt.
        Args:
            prompt (str): Prompt to generate text from.
            max_length (int): Maximum length of the response.
        """
        # Encode the prompt text to be compatible with the model
        encoded_input = self.tokenizer.encode(prompt, return_tensors='pt')
        encoded_input = encoded_input.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate a sequence of tokens in response to the input
        output_sequences = self.model.generate(
            input_ids=encoded_input,
            max_length=max_length + len(encoded_input[0]),
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        
        # Decode the generated sequence to a readable string
        return self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)

