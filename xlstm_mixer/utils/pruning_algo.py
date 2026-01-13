# xLSTM-Mixer/xlstm_mixer/utils/pruning_algo.py
import torch
import math

@torch.no_grad()
def get_fss_mask(weight: torch.Tensor, bank_num: int, sparsity: float) -> torch.Tensor:
    """
    Algorithm 1: FSS Pruning Logic (Bank-Balanced Sparsity)
    
    Args:
        weight: (Out_Features, In_Features) Weight matrix
        bank_num: Number of banks per row
        sparsity: Target sparsity (0.0 - 1.0)
    Returns:
        mask: Bool tensor (True=KEEP, False=PRUNE)
    """
    rows, cols = weight.shape
    
    # --- Constraint Check 1: Divisibility ---
    if cols % bank_num != 0:
        raise ValueError(
            f"FSS Error: In_Features ({cols}) must be divisible by Bank_Num ({bank_num}). "
            f"Module shape: {weight.shape}. Please adjust bank_num."
        )
    
    bank_size = cols // bank_num
    
    # --- Constraint Check 2: NNZ Consistency ---
    keep_k = int(round(bank_size * (1 - sparsity)))
    if keep_k < 1: 
        keep_k = 1 
    if keep_k > bank_size:
        keep_k = bank_size

    # --- Step 1: Calculate Score (Magnitude Pruning) ---
    score = torch.abs(weight)
    
    # --- Step 2: Reshape to Banks ---
    w_reshaped = score.reshape(rows, bank_num, bank_size)
    
    # --- Step 3: Top-K Selection within each Bank ---
    _, indices = torch.topk(w_reshaped, k=keep_k, dim=2, largest=True, sorted=False)
    
    # --- Step 4: Generate Mask (Bool) ---
    mask_reshaped = torch.zeros_like(w_reshaped, dtype=torch.bool)
    mask_reshaped.scatter_(2, indices, True)
    
    # Restore shape
    mask = mask_reshaped.reshape(rows, cols)
    
    return mask

@torch.no_grad()
def check_fss_compliance(weight: torch.Tensor, mask: torch.Tensor, bank_num: int):
    """
    Strict Verification tool. Raises AssertionError if FSS constraints are violated.
    """
    rows, cols = weight.shape
    bank_size = cols // bank_num
    
    # 1. Strict Mask persistence check (Exact Zero)
    pruned_weights = weight[~mask]
    if pruned_weights.numel() > 0 and (pruned_weights != 0).any():
        raise AssertionError("FSS Violation: Non-zero weight found in masked region!")

    # 2. Bank balance check
    # Count non-zeros directly from bool mask (returns int/long)
    mask_reshaped = mask.reshape(rows, bank_num, bank_size)
    nnz_counts = mask_reshaped.sum(dim=2) # [rows, bank_num]
    
    # Check intra-row consistency
    ref_nnz_per_row = nnz_counts[:, :1] # [rows, 1]
    
    if not (nnz_counts == ref_nnz_per_row).all():
        min_nnz = nnz_counts.min().item()
        max_nnz = nnz_counts.max().item()
        raise AssertionError(
            f"FSS Violation: Banks are NOT balanced within rows! "
            f"NNZ range across all banks: [{min_nnz}, {max_nnz}]. "
        )
    
    return {
        "nnz_per_bank_sample": int(ref_nnz_per_row[0].item()),
        "total_params": weight.numel(),
        "total_nnz": int(nnz_counts.sum().item())
    }
