import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertTokenizer, BertForMaskedLM

class BERTTextGenerator:
    def __init__(self):
        # Load pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased', attn_implementation="eager")
        
    def generate_text(self, template, num_suggestions=3):
        """
        Generate text by replacing masked tokens
        """
        inputs = self.tokenizer(template, return_tensors='pt')
        mask_token_index = torch.where(inputs['input_ids'] == self.tokenizer.mask_token_id)[1]
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = outputs.logits
        mask_token_logits = predictions[0, mask_token_index, :]
        top_tokens = torch.topk(mask_token_logits, num_suggestions, dim=1)
        
        results = []
        for i in range(num_suggestions):
            tokens = top_tokens.indices[0, i].unsqueeze(0)
            word = self.tokenizer.decode(tokens)
            probability = torch.softmax(top_tokens.values[0], dim=0)[i].item()
            filled_text = template.replace('[MASK]', word)
            results.append({
                'text': filled_text,
                'probability': probability
            })
        return results
    
    def visualize_word_importance(self, sentence):
        try:
            # Tokenize the input
            encoded = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )

            # Get model outputs with attention
            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    output_attentions=True
                )

            # Check if attentions are returned properly
            if outputs.attentions is None:
                raise ValueError("Attention outputs are missing from the model output")

            # Get attention weights
            # Shape: [layers, batch, heads, seq_length, seq_length]
            attention_tensor = torch.stack(outputs.attentions)
            
            # Average across layers and heads to get [seq_length, seq_length]
            avg_attention = torch.mean(attention_tensor, dim=[0, 1, 2]).squeeze(0)

            # Convert to numpy for visualization
            attention_matrix = avg_attention.numpy()

            # Get tokens for labels
            tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

            # Create heatmap
            plt.figure(figsize=(12, 8), dpi=100)
            sns.heatmap(
                attention_matrix,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='YlOrRd',
                square=True,
                cbar_kws={'label': 'Attention Weight'}
            )

            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.title('BERT Attention Visualization', pad=20)
            plt.xlabel('Target Tokens', labelpad=10)
            plt.ylabel('Source Tokens', labelpad=10)
            plt.tight_layout()

            # Save to buffer and encode as base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return image_base64

        except Exception as e:
            print(f"Error in visualize_word_importance: {e}")
            raise


# Pydantic
class TextGenerationRequest(BaseModel):
    template: str
    num_suggestions: int = 3

class AttentionVisualizationRequest(BaseModel):
    sentence: str

# FastAPI
app = FastAPI(
    title="BERT Text Generation API",
    description="API for BERT-based text generation and attention visualization",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize BERT generator
bert_generator = BERTTextGenerator()

@app.post("/generate-text/")
async def generate_text(request: TextGenerationRequest):
    #text suggestions
    try:
        results = bert_generator.generate_text(
            request.template, 
            request.num_suggestions
        )
        return {"suggestions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualize-attention/")
async def visualize_attention(request: AttentionVisualizationRequest):
    #attention heatmap 
    
    try:
        image_base64 = bert_generator.visualize_word_importance(request.sentence)
        return {"attention_heatmap": image_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    #health checkpoint
    return {
        "message": "BERT Text Generation API is running",
        "available_endpoints": [
            "/generate-text/",
            "/visualize-attention/"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)