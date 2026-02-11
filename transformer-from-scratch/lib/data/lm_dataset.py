from typing import Tuple, List
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
from .tokenizer import H4Tokenizer


    
class LMDataset(Dataset):
    """
    Dataset for Language Model training/evaluation.
    """
    def __init__(
            self, 
            partition: str, 
            config: dict, 
            tokenizer: H4Tokenizer
    ):
        """
        Initializes the Language Model Dataset for training language models on text data.

        Args:
            partition (str): Data partition subdirectory under root (e.g., 'train', 'test')
            config (dict): Configuration dictionary containing dataset settings
            tokenizer (H4Tokenizer): Tokenizer for encoding/decoding text
        """
        
        # Store configuration and other args
        self.config    = config
        self.partition = partition
        self.tokenizer = tokenizer

        self.eos_token = self.tokenizer.eos_id
        self.sos_token = self.tokenizer.sos_id
        self.pad_token = self.tokenizer.pad_id

        # Set up data paths 
        root = self.config.get("root", os.path.join("data_subset", "p1_data"))
        self.text_dir = os.path.join(root, partition)

        self.text_files = sorted([os.path.join(self.text_dir, f) for f in os.listdir(self.text_dir) if f.endswith(".npy")])

        subset_size =  int(len(self.text_files) * self.config["subset"]) 
        self.text_files = self.text_files[:subset_size]

        # Initialize lists to store transcripts
        self.transcripts_shifted = []
        self.transcripts_golden  = []
        
        # Initialize tracking variables
        self.total_chars  = 0
        self.total_tokens = 0
        self.text_max_len = 0
        
        print(f"Loading transcripts for {partition} partition...")
        for file in tqdm(self.text_files):
            transcript = "".join(np.load(file, allow_pickle=True).tolist())
            
            # Track character count (before tokenization)
            self.total_chars += len(transcript)
            
            tokenized = self.tokenizer.encode(transcript)
            
            # Track token count (excluding special tokens)
            self.total_tokens += len(tokenized)
            
            # Track max length (add 1 for the sos/eos tokens)
            self.text_max_len = max(self.text_max_len, len(tokenized)+1)
            
            self.transcripts_shifted.append([self.sos_token] + tokenized)
            self.transcripts_golden.append(tokenized + [self.eos_token])

        # Calculate average characters per token
        self.avg_chars_per_token = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        
        # Verify data alignment
        if not (len(self.transcripts_shifted) == len(self.transcripts_golden)):
            raise ValueError("Shifted and golden transcripts are misaligned")
        
        self.length = len(self.transcripts_shifted)
        
    def get_avg_chars_per_token(self) -> float:
        '''
        Get the average number of characters per token. Used to calculate character-level perplexity.
        '''
        return self.avg_chars_per_token
    
    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (shifted_transcript, golden_transcript) where:
                - shifted_transcript: LongTensor starting with SOS token
                - golden_transcript: LongTensor ending with EOS token
        """
        shifted = torch.LongTensor(self.transcripts_shifted[idx])
        golden  = torch.LongTensor(self.transcripts_golden[idx])
        return shifted, golden
    
    
    def collate_fn(self, batch: List[Tuple[torch.LongTensor, torch.LongTensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate and pad a batch of samples to create a batch of fixed-length padded shifted and golden transcripts.

        Args:
            batch (list): List of (shifted, golden) transcript pairs

        Returns:
            tuple: (padded_shifted, padded_golden, lengths) where:
                - padded_shifted: Tensor of shape (batch, max_len) with SOS prefixes
                - padded_golden: Tensor of shape (batch, max_len) with EOS suffixes
                - lengths: Original sequence lengths before padding
        """
        shifted_transcripts, golden_transcripts = zip(*batch)
        
        lengths = torch.LongTensor([t.size(0) for t in shifted_transcripts]) # (B)

        padded_shifted = pad_sequence(shifted_transcripts, batch_first=True, padding_value=self.pad_token) # (B, T)
        padded_golden  = pad_sequence(golden_transcripts,  batch_first=True, padding_value=self.pad_token) # (B, T)

        return padded_shifted, padded_golden, lengths

    def sample_prompts(self, num_samples: int, prompt_length: int, seed: int = None) -> Tuple[torch.LongTensor, List[torch.LongTensor]]:
        """
        Sample random prompts of fixed length from the dataset and return their original sequences.
        '''
        
        Args:
            num_samples: Number of prompts to sample
            prompt_length: Exact number of tokens for each prompt
            seed: Random seed for reproducibility. If None, no seed is set.
            
        Returns:
            tuple: (prompts, originals) where:
                - prompts: torch.LongTensor of tokenized prompts
                - originals: List of torch.LongTensor containing complete original sequences
        """
        if seed is not None:
            np_state = np.random.get_state()
            np.random.seed(seed)
            
        prompts = []
        originals = []
        attempts = 0
        max_attempts = num_samples * 10  # Prevent infinite loops
        
        while len(prompts) < num_samples and attempts < max_attempts:
            idx = np.random.randint(0, len(self))
            tokens = self.transcripts_shifted[idx][1:] # remove sos token
            
            if len(tokens) < prompt_length:
                attempts += 1
                continue
                
            prompt_tokens = tokens[:prompt_length]
            
            prompts.append(torch.LongTensor([self.sos_token] + prompt_tokens))
            originals.append(torch.LongTensor(tokens + [self.eos_token]))
            
            attempts += 1
            
        if len(prompts) < num_samples:
            print(f"Warning: Could only sample {len(prompts)} valid prompts")
        
        if seed is not None:
            np.random.set_state(np_state)
            
        return torch.stack(prompts), originals