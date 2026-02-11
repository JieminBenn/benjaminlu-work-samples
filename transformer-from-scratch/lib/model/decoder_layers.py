import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, CrossAttentionLayer, FeedForwardLayer



## -------------------------------------------------------------------------------------------------  
## Decoder Layers
## -------------------------------------------------------------------------------------------------      
class SelfAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer with masked self-attention and feed-forward sublayers.
    Used in the decoder-only Transformer architecture.  
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Initialize the SelfAttentionDecoderLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        ''' 
        super().__init__()
       
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout) # Masked self-attention layer
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout) # Feed-forward network

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the DecoderLayer1.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)   
            key_padding_mask (torch.Tensor): The padding mask for the decoder. shape: (batch_size, seq_len)
            attn_mask (torch.Tensor): The self-attention mask. shape: (seq_len, seq_len)

        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, num_classes)
            mha_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)   
        '''

        x, mha_attn_weights = self.self_attn(x,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
        x = self.ffn(x)
        
        return x, mha_attn_weights 

## -------------------------------------------------------------------------------------------------    
class CrossAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer with masked self-attention, cross-attention, and feed-forward sublayers.
    Used in the encoder-decoder Transformer architecture.
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Initialize the CrossAttentionDecoderLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        '''
        super().__init__()

        self.self_attn  = SelfAttentionLayer(d_model, num_heads, dropout) # Masked self-attention layer
        self.cross_attn = CrossAttentionLayer(d_model, num_heads, dropout) # Cross-attention layer
        self.ffn        = FeedForwardLayer(d_model, d_ff, dropout) # Feed-forward network

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, dec_key_padding_mask: Optional[torch.Tensor] = None, enc_key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the CrossAttentionDecoderLayer.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)   
            enc_output (torch.Tensor): The encoder output. shape: (batch_size, seq_len, d_model)
            dec_key_padding_mask (Optional[torch.Tensor]): The padding mask for the decoder input. shape: (batch_size, seq_len)
            enc_key_padding_mask (Optional[torch.Tensor]): The padding mask for the encoder output. shape: (batch_size, seq_len')
            attn_mask (Optional[torch.Tensor]): The self-attention mask for the decoder input. shape: (seq_len, seq_len)
        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, d_model)
            self_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)   
            cross_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)    
        '''

        x, self_attn_weights  = self.self_attn(x, key_padding_mask=dec_key_padding_mask, attn_mask=attn_mask)
        x, cross_attn_weights = self.cross_attn(x, enc_output, key_padding_mask=enc_key_padding_mask, attn_mask=None)
        x = self.ffn(x)
        return x, self_attn_weights, cross_attn_weights 
## -------------------------------------------------------------------------------------------------    
