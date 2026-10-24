"""
CNN-LSTM Model for Image Captioning (Baseline)
Encoder: ResNet50
Decoder: 2-layer LSTM
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):
    """CNN Encoder using ResNet50"""
    
    def __init__(self, embed_size=512, pretrained=True):
        super(EncoderCNN, self).__init__()

        # Load ResNet50 (ImageNet weights for training; when loading a checkpoint the
        # weights come from the checkpoint, so nothing needs to be downloaded)
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Remove the last fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Freeze ResNet parameters (transfer learning)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Linear layer to project to embedding space
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        
    def forward(self, images, track_grad=False):
        """Extract feature vectors from images.

        The frozen ResNet runs without gradients during training. Grad-CAM needs
        gradients with respect to the ResNet feature maps, so it passes track_grad=True.
        """
        with torch.set_grad_enabled(track_grad):
            features = self.resnet(images)  # (batch, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, 2048)
        features = self.fc(features)  # (batch, embed_size)
        features = self.bn(features)
        return features

class DecoderLSTM(nn.Module):
    """LSTM Decoder for caption generation"""
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.5):
        super(DecoderLSTM, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Word embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embed_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions, lengths):
        """Teacher forcing with correct next-token alignment.
        captions: [<START>, w1, w2, ..., <END>, <PAD>...]
        We want to predict: [w1, w2, ..., <END>]
        """
        # Drop <START> so targets begin at w1 (this matches your captioning_loss targets=captions[:,1:])
        embeddings = self.embed(captions[:, 1:])  # (B, L-1, E)

        # Inputs should be: [features] + [w1..w_{L-2}]  -> total length L-1
        features = features.unsqueeze(1)  # (B, 1, E)
        embeddings = torch.cat((features, embeddings[:, :-1, :]), dim=1)  # (B, L-1, E)

        # Lengths must match the sequence length we feed (L-1)
        if isinstance(lengths, torch.Tensor):
            lengths = (lengths - 1).clamp(min=1).cpu()
        else:
            lengths = [max(l - 1, 1) for l in lengths]

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)

        hiddens, _ = self.lstm(packed)

        # Unpack (keeps variable-length sequences)
        hiddens, _ = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)

        hiddens = self.dropout(hiddens)
        outputs = self.fc(hiddens)  # (B, L-1, vocab_size)

        return outputs

    
    def generate_caption(self, features, vocab, max_length=40, beam_size=3):
        """Generate caption using beam search"""
        batch_size = features.size(0)
        
        if beam_size == 1:
            return self._greedy_search(features, vocab, max_length)
        else:
            return self._beam_search(features, vocab, max_length, beam_size)
    
    def _greedy_search(self, features, vocab, max_length):
        """Generate caption using greedy search"""
        captions = []
        
        # Initialize hidden state
        states = None
        inputs = features.unsqueeze(1)  # (1, 1, embed_size)
        
        for _ in range(max_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.fc(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            
            captions.append(predicted.item())
            
            # Stop if END token is generated
            if predicted.item() == vocab.word2idx['<END>']:
                break
            
            # Prepare next input
            inputs = self.embed(predicted).unsqueeze(1)
        
        # Convert indices to words
        caption_words = [vocab.idx2word[idx] for idx in captions 
                        if idx not in [vocab.word2idx['<START>'], vocab.word2idx['<END>'], vocab.word2idx['<PAD>']]]
        
        return ' '.join(caption_words)
    
    def _beam_search(self, features, vocab, max_length, beam_size):
        """Generate caption using beam search"""
        # Start with START token
        start_idx = vocab.word2idx['<START>']
        end_idx = vocab.word2idx['<END>']
        
        # Initialize beam
        # Each beam element: (sequence, score, hidden_state)
        beams = [(torch.LongTensor([start_idx]).to(features.device), 0.0, None)]
        completed = []
        
        inputs = features.unsqueeze(1)  # (1, 1, embed_size)
        
        for step in range(max_length):
            candidates = []
            
            for seq, score, states in beams:
                # If sequence already ended, add to candidates
                if seq[-1].item() == end_idx:
                    completed.append((seq, score))
                    continue
                
                # Get last token embedding
                if step == 0:
                    lstm_input = inputs
                else:
                    lstm_input = self.embed(seq[-1]).unsqueeze(0).unsqueeze(0)
                
                # LSTM forward
                hiddens, new_states = self.lstm(lstm_input, states)
                outputs = self.fc(hiddens.squeeze(1))
                log_probs = torch.log_softmax(outputs, dim=1)
                
                # Get top k predictions
                top_log_probs, top_indices = log_probs.topk(beam_size)
                
                for log_prob, idx in zip(top_log_probs[0], top_indices[0]):
                    new_seq = torch.cat([seq, idx.unsqueeze(0)])
                    new_score = score + log_prob.item()
                    candidates.append((new_seq, new_score, new_states))
            
            # Select top beam_size candidates
            if candidates:
                beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # Stop if all beams ended
            if all(seq[-1].item() == end_idx for seq, _, _ in beams):
                break
        
        # Get best sequence
        all_sequences = completed + [(seq, score) for seq, score, _ in beams]
        best_seq, _ = max(all_sequences, key=lambda x: x[1])
        
        # Convert to words
        caption_words = [vocab.idx2word[idx.item()] for idx in best_seq 
                        if idx.item() not in [vocab.word2idx['<START>'], vocab.word2idx['<END>'], vocab.word2idx['<PAD>']]]
        
        return ' '.join(caption_words)

class CNNLSTMModel(nn.Module):
    """Complete CNN-LSTM model for image captioning"""
    
    def __init__(self, embed_size=512, hidden_size=512, vocab_size=8256, num_layers=2, dropout=0.5,
                 pretrained=True):
        super(CNNLSTMModel, self).__init__()

        self.encoder = EncoderCNN(embed_size, pretrained=pretrained)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers, dropout)
        
    def forward(self, images, captions, lengths):
        """Forward pass"""
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs
    
    def generate_caption(self, image, vocab, max_length=40, beam_size=3):
        """Generate caption for a single image"""
        self.eval()
        with torch.no_grad():
            features = self.encoder(image.unsqueeze(0))
            caption = self.decoder.generate_caption(features, vocab, max_length, beam_size)
        return caption
    
    def save_checkpoint(self, filepath, vocab, optimizer=None, epoch=None, loss=None):
        """Save model checkpoint"""
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'vocab': vocab,
            'embed_size': self.decoder.embed_size,
            'hidden_size': self.decoder.hidden_size,
            'vocab_size': self.decoder.vocab_size,
            'num_layers': self.decoder.num_layers,
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch:
            checkpoint['epoch'] = epoch
        if loss:
            checkpoint['loss'] = loss
        
        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint saved to {filepath}")
    
    @classmethod
    def load_from_checkpoint(cls, filepath, device='cpu'):
        """Load model from checkpoint.

        The checkpoint stores the Vocabulary object, so it is loaded with
        weights_only=False. Only load checkpoints you trust.
        Weights saved in float16 (see slim_checkpoint.py) are cast back to float32.
        """
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        model = cls(
            embed_size=checkpoint['embed_size'],
            hidden_size=checkpoint['hidden_size'],
            vocab_size=checkpoint['vocab_size'],
            num_layers=checkpoint['num_layers'],
            pretrained=False,
        )

        def to_float32(state_dict):
            return {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}

        model.encoder.load_state_dict(to_float32(checkpoint['encoder_state_dict']))
        model.decoder.load_state_dict(to_float32(checkpoint['decoder_state_dict']))
        model.to(device)
        model.eval()

        return model, checkpoint.get('vocab'), checkpoint.get('epoch'), checkpoint.get('loss')
