import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
import matplotlib.font_manager as fm
from langdetect import detect
import re

class BilingualBERTGenerator:
    def __init__(self):
        # Load XLM-RoBERTa large for better performance on both languages
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large", attn_implementation="eager")
        
        # Initialize Vietnamese-specific model
        self.vn_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.vn_model = AutoModelForMaskedLM.from_pretrained("vinai/phobert-base", attn_implementation="eager")
        
        # Define mask patterns
        self.mask_patterns = [
            '_____',          # Basic 5 underscores
            r'\[___\]',      # [___]
            r'\{...\}',      # {...}
            r'\[MASK\]',     # [MASK]
            '<mask>'         # <mask>
        ]
        
    def standardize_mask(self, text):
        """Convert any mask pattern to the model's mask token"""
        for pattern in self.mask_patterns:
            text = re.sub(pattern, self.tokenizer.mask_token, text)
        return text
    
    def restore_mask_format(self, text, original_format):
        """Restore the original mask format in the generated text"""
        return text.replace(self.tokenizer.mask_token, original_format)
    
    def detect_language(self, text):
        try:
            # Remove all possible mask patterns for language detection
            clean_text = text
            for pattern in self.mask_patterns:
                clean_text = re.sub(pattern, '', clean_text)
            return detect(clean_text.strip())
        except:
            return 'en'
    
    def find_original_mask(self, text):
        """Find the original mask format used in the text"""
        for pattern in self.mask_patterns:
            if re.search(pattern, text):
                match = re.search(pattern, text)
                return match.group(0)
        return '_____'  # default mask if none found
    
    def generate_text(self, template, num_suggestions=3):
        # Store original mask format
        original_mask = self.find_original_mask(template)
        
        # Standardize mask token
        template = self.standardize_mask(template)
        
        # Detect language
        lang = self.detect_language(template)
        
        # Choose appropriate model based on language
        current_model = self.vn_model if lang == 'vi' else self.model
        current_tokenizer = self.vn_tokenizer if lang == 'vi' else self.tokenizer
        
        # Add spaces for Vietnamese text
        if lang == 'vi':
            template = ' '.join(template.split())
        
        inputs = current_tokenizer(template, return_tensors='pt')
        mask_token_index = torch.where(inputs['input_ids'] == current_tokenizer.mask_token_id)[1]
        
        with torch.no_grad():
            outputs = current_model(**inputs)
            
        predictions = outputs.logits
        mask_token_logits = predictions[0, mask_token_index, :]
        top_tokens = torch.topk(mask_token_logits, num_suggestions, dim=1)
        
        results = []
        for i in range(num_suggestions):
            tokens = top_tokens.indices[0, i].unsqueeze(0)
            word = current_tokenizer.decode(tokens).strip()
            probability = torch.softmax(top_tokens.values[0], dim=0)[i].item()
            filled_text = template.replace(current_tokenizer.mask_token, word)
            
            # Restore original mask format in the text
            original_format_text = filled_text
            
            results.append({
                'text': original_format_text,
                'word': word,
                'probability': probability,
                'detected_language': lang
            })
        return results
    
    def visualize_word_importance(self, sentence):
        try:
            # Detect language
            lang = self.detect_language(sentence)
            
            # Choose appropriate model based on language
            current_model = self.vn_model if lang == 'vi' else self.model
            current_tokenizer = self.vn_tokenizer if lang == 'vi' else self.tokenizer
            
            # Set up font for multi-language support
            plt.rcParams['font.family'] = 'DejaVu Sans'
            
            # Process Vietnamese text specifically
            if lang == 'vi':
                sentence = ' '.join(sentence.split())
            
            encoded = current_tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = current_model(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    output_attentions=True
                )

            attention_tensor = torch.stack(outputs.attentions)
            
            # Use last 4 layers for more meaningful patterns
            last_4_layers = attention_tensor[-4:]
            avg_attention = torch.mean(last_4_layers, dim=[0, 1, 2]).squeeze(0)
            attention_matrix = avg_attention.numpy()

            # Get tokens and clean up display
            tokens = current_tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
            # Clean up token display
            cleaned_tokens = [t.replace('▁', '') for t in tokens]
            
            plt.figure(figsize=(12, 8), dpi=100)
            sns.heatmap(
                attention_matrix,
                xticklabels=cleaned_tokens,
                yticklabels=cleaned_tokens,
                cmap='YlOrRd',
                square=True,
                cbar_kws={'label': 'Attention Weight'}
            )

            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            model_name = "PhoBERT" if lang == 'vi' else "XLM-RoBERTa"
            plt.title(f'{model_name} Attention Visualization ({lang.upper()})', pad=20)
            plt.xlabel('Target Tokens', labelpad=10)
            plt.ylabel('Source Tokens', labelpad=10)
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return {
                'image': image_base64,
                'detected_language': lang
            }

        except Exception as e:
            print(f"Error in visualize_word_importance: {e}")
            raise

# Pydantic models
class TextGenerationRequest(BaseModel):
    template: str
    num_suggestions: int = 3
    
    class Config:
        schema_extra = {
            "example": {
                "template": "Today the weather is _____",
                "num_suggestions": 3
            }
        }

class AttentionVisualizationRequest(BaseModel):
    sentence: str

# FastAPI setup
app = FastAPI(
    title="Bilingual BERT Text Generation API",
    description="""
    API for Vietnamese and English text generation and attention visualization.
    Supported mask formats:
    - _____ (5 underscores)
    - [___]
    - {...}
    - [MASK]
    - <mask>
    """,
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize generator
bert_generator = BilingualBERTGenerator()

@app.post("/generate-text/")
async def generate_text(request: TextGenerationRequest):
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
    try:
        result = bert_generator.visualize_word_importance(request.sentence)
        return {
            "attention_heatmap": result['image'],
            "detected_language": result['detected_language']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mask-formats")
async def get_mask_formats():
    """Get information about supported mask formats"""
    return {
        "supported_formats": [
            {
                "format": "_____",
                "description": "Five underscores (recommended)",
                "example": "The weather is _____ today"
            },
            {
                "format": "[___]",
                "description": "Three underscores in brackets",
                "example": "The weather is [___] today"
            },
            {
                "format": "{...}",
                "description": "Three dots in curly braces",
                "example": "The weather is {...} today"
            },
            {
                "format": "[MASK]",
                "description": "Traditional BERT mask",
                "example": "The weather is [MASK] today"
            },
            {
                "format": "<mask>",
                "description": "XML-style mask",
                "example": "The weather is <mask> today"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)