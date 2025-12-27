import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from models.fusion_detector import FusionDetector

def apply_pruning(model, amount=0.3):
    """
    Prunes a percentage of weights in Conv2d and Linear layers
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # Make pruning permanent
            """
            Note:
            This rule applies everywhere:
            Prune → remove reparameterization → save → quantize → export.
            When I applied pruning without removing the re-parametrization,
            the model size reduced but rewrote the layer like  "weight = weight_orig * weight_mask"
            and hence the quantization step failed later on. Thus the prune.remove(module, 'weight') is necessary.
            """
    return model

if __name__ == "__main__":
    # Initialize model
    model = FusionDetector()
    model.load_state_dict(torch.load("checkpoints/fusion_model.pt", map_location='cpu'))

    print("Before pruning:", 
        sum(p.numel() for p in model.parameters()))

    # Apply pruning
    model = apply_pruning(model, amount=0.3)

    print("After pruning:", 
        sum(p.numel() for p in model.parameters()))
    
    # Save the pruned model
    torch.save(model.state_dict(), "checkpoints/fusion_model_pruned.pt")
    print("Pruned model saved.")