import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class T5Chat:
    """Model for generating responses using T5."""
    def __init__(self, model_name='t5-small'):
        """Initialize the model and tokenizer.
        Args:
            model_name (str): Name of the T5 model to use.
        """
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    def generate_response(self, prompt, max_length=150):
        """Generate a text response based on the input prompt.
        Args:
            prompt (str): Prompt to generate text from.
            max_length (int): Maximum length of the response.
        """
        # Prepare the prompt text for T5 which expects a certain task-related format
        input_text = "translate English to English: " + prompt

        # Encode the prompt text
        encoded_input = self.tokenizer.encode(input_text, return_tensors='pt')
        encoded_input = encoded_input.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Generate tokens
        output_sequences = self.model.generate(
            input_ids=encoded_input,
            max_length=max_length,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )

        # Decode the generated tokens to text
        return self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)

