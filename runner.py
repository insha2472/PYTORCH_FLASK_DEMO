"""
Text Generator Runner
=====================
Loads and runs the text generator model.
"""

import torch
import torch.nn as nn


class TextGeneratorLSTM(nn.Module):
    """LSTM-based Text Generator Model"""
    
    def __init__(self, vocab_size=5273, embedding_dim=16, hidden_size=32):
        super(TextGeneratorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        if hidden is None:
            output, hidden = self.lstm(embedded)
        else:
            output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden
    
    def init_hidden(self, batch_size=1):
        device = next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return (h0, c0)


def load_model(model_path='text generator.pth', device='cpu'):
    """Load the trained model"""
    model = TextGeneratorLSTM()
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def generate_text(model, start_indices, num_words=100, temperature=1.0):
    """
    Generate text (as indices) from the model.
    
    Args:
        model: Trained model
        start_indices: List of starting token indices
        num_words: Number of tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
    
    Returns:
        List of generated token indices
    """
    model.eval()
    device = next(model.parameters()).device
    
    generated = list(start_indices)
    hidden = model.init_hidden(batch_size=1)
    
    with torch.no_grad():
        # Process seed tokens
        for idx in start_indices[:-1]:
            x = torch.tensor([[idx]]).to(device)
            _, hidden = model(x, hidden)
        
        # Generate new tokens
        current_idx = start_indices[-1] if start_indices else 0
        
        for _ in range(num_words):
            x = torch.tensor([[current_idx]]).to(device)
            output, hidden = model(x, hidden)
            
            logits = output[0, -1] / temperature
            probs = torch.softmax(logits, dim=0)
            current_idx = torch.multinomial(probs, 1).item()
            generated.append(current_idx)
    
    return generated


if __name__ == '__main__':
    print("="*60)
    print("        TEXT GENERATOR - PyTorch LSTM Model")
    print("="*60)
    
    # Load model
    model = load_model('text generator.pth')
    print("✓ Model loaded successfully!\n")
    
    # Demo: Generate tokens starting from random indices
    start_tokens = [100, 200]  # Example starting indices
    
    print("Generating 100 tokens...")
    generated_indices = generate_text(model, start_tokens, num_words=100, temperature=0.8)
    
    print(f"\n✓ Generated {len(generated_indices)} tokens")
    print(f"First 20 token indices: {generated_indices[:20]}")
    print(f"Last 20 token indices: {generated_indices[-20:]}")
    
    print("\n" + "="*60)
    print("Model is working! To convert to text, provide vocabulary.")
    print("="*60)