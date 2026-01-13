from lightning.pytorch.callbacks import Callback
import torch
import torch.nn as nn
from ..utils.pruning_algo import get_fss_mask, check_fss_compliance

class FSSPruningCallback(Callback):
    def __init__(
        self, 
        target_sparsity: float = 0.0, # Default 0 = disabled
        bank_num: int = 4, 
        bank_size: int = None, # If set, overrides bank_num dynamically per layer
        start_epoch: int = 0,
        warmup_epochs: int = 10,
        pruning_frequency: int = 1,
        target_modules: list[str] = None
    ):
        """
        Robust FSS Pruning Callback.
        - Supports sLSTM 3D parameters (Heads, Out, In)
        - Supports standard nn.Linear
        - Uses Bool Masks and masked_fill_ for stability.
        """
        super().__init__()
        self.target_sparsity = target_sparsity
        self.bank_num = bank_num
        self.bank_size = bank_size # Store bank_size preference
        self.start_epoch = start_epoch
        self.warmup_epochs = warmup_epochs
        self.pruning_frequency = pruning_frequency
        
        if target_modules is None:
            self.target_keywords = ["igate", "fgate", "ogate", "zgate", "_recurrent_kernel_"]
        else:
            self.target_keywords = target_modules
        
        self.exclude_keywords = ["proj_up", "proj_down", "bias", "norm"] # Explicitly exclude FFN
            
        self.current_sparsity = 0.0
        # pruned_modules stores: name -> (parent_module, param_name, mask_name, is_3d)
        self.pruned_params = {} 
        self._last_global_step = -1

    def _is_target_param(self, name):
        if "xlstm.blocks" not in name: return False
        if any(k in name for k in self.exclude_keywords): return False
        if not any(k in name for k in self.target_keywords): return False
        return True

    def setup(self, trainer, pl_module, stage=None):
        if self.target_sparsity <= 0:
            return

        if stage in (None, "fit"):
            count = 0
            # Scan parameters instead of modules
            for name, param in pl_module.named_parameters():
                if not param.requires_grad: continue
                
                if self._is_target_param(name):
                    # name is like 'model.xlstm.blocks.0.xlstm.fgate.weight'
                    # parent module name is 'model.xlstm.blocks.0.xlstm.fgate'
                    # param name is 'weight'
                    
                    split_name = name.rsplit('.', 1)
                    if len(split_name) != 2: continue
                    parent_name, param_attr = split_name
                    
                    # Find parent module
                    parent_module = pl_module
                    if parent_name:
                        for part in parent_name.split('.'):
                            if part == "model" and hasattr(pl_module, "model"): 
                                # Handle LightningModule wrapper if needed, 
                                # but named_parameters usually handles it.
                                # pl_module.named_parameters() keys start from pl_module's children.
                                # If pl_module has 'model', keys might be 'model.xlstm...'
                                # If 'part' is an attribute of 'parent_module', get it.
                                if hasattr(parent_module, part):
                                    parent_module = getattr(parent_module, part)
                                else:
                                    # Fallback for some weird naming or direct access
                                    pass
                            elif hasattr(parent_module, part):
                                parent_module = getattr(parent_module, part)
                            elif part.isdigit(): # Sequential or ModuleList
                                parent_module = parent_module[int(part)]
                            else:
                                # Could not traverse
                                print(f"[FSS Warning] Could not find parent module for {name}")
                                parent_module = None
                                break
                    
                    if parent_module is None: continue

                    # Check dimensions
                    # Case 1: 3D [Heads, Out, In] -> [8, 32, 32] or [8, 32, 128]
                    # Case 2: 2D [Out, In] -> standard Linear
                    
                    is_3d = False
                    if param.dim() == 3:
                        is_3d = True
                        heads, d_out, d_in = param.shape
                        pruning_dim = d_in # Usually the last dim is input features
                    elif param.dim() == 2:
                        d_out, d_in = param.shape
                        pruning_dim = d_in
                    else:
                        continue # Skip 1D biases or other shapes

                    # Determine Bank Num (Fixed Num vs Fixed Size)
                    if self.bank_size is not None:
                        # Mode 2: Fixed Bank Size (e.g. 4 for 2:4 sparsity)
                        if pruning_dim % self.bank_size != 0:
                            print(f"[FSS Skip] {name}: dim ({pruning_dim}) % bank_size ({self.bank_size}) != 0")
                            continue
                        local_bank_num = pruning_dim // self.bank_size
                        if local_bank_num < 1: continue
                    else:
                        # Mode 1: Fixed Bank Num (Original FSS)
                        if pruning_dim % self.bank_num != 0:
                            print(f"[FSS Skip] {name}: pruning dim ({pruning_dim}) % {self.bank_num} != 0")
                            continue
                        local_bank_num = self.bank_num

                    mask_name = f"{param_attr}_mask"
                    
                    # Register Buffer
                    if not hasattr(parent_module, mask_name):
                        mask = torch.ones_like(param, dtype=torch.bool)
                        parent_module.register_buffer(mask_name, mask)
                    
                    # Store local_bank_num for later use
                    self.pruned_params[name] = (parent_module, param_attr, mask_name, is_3d, local_bank_num)
                    count += 1
                    
                    # Initial Apply
                    with torch.no_grad():
                        mask = getattr(parent_module, mask_name)
                        param.masked_fill_(~mask, 0.0)

            print(f"[FSS Setup] Managing FSS masks for {len(self.pruned_params)} parameters. Target Sparsity: {self.target_sparsity}")
            if self.bank_size:
                print(f"[FSS Config] Mode: Fixed Bank Size = {self.bank_size} (Dynamic Bank Num per layer)")
            else:
                print(f"[FSS Config] Mode: Fixed Bank Num = {self.bank_num}")

    def on_train_epoch_start(self, trainer, pl_module):
        if self.target_sparsity <= 0: return

        current_epoch = trainer.current_epoch
        if current_epoch < self.start_epoch: return

        if current_epoch >= self.start_epoch + self.warmup_epochs:
            self.current_sparsity = self.target_sparsity
        else:
            t = current_epoch - self.start_epoch
            T = self.warmup_epochs
            self.current_sparsity = self.target_sparsity * (1 - (1 - t/T)**3)

        if (current_epoch - self.start_epoch) % self.pruning_frequency == 0:
            if self.current_sparsity > 0:
                print(f"\n[FSS Pruning] Epoch {current_epoch}: Updating masks to sparsity {self.current_sparsity:.4f}")
                self._update_masks()

    def _update_masks(self):
        for name, (parent_module, param_attr, mask_name, is_3d, local_bank_num) in self.pruned_params.items():
            param = getattr(parent_module, param_attr)
            mask_buffer = getattr(parent_module, mask_name)
            
            weight_to_prune = param.detach()
            
            if is_3d:
                # Reshape 3D [H, O, I] -> 2D [H*O, I]
                heads, d_out, d_in = weight_to_prune.shape
                weight_flat = weight_to_prune.reshape(-1, d_in)
                new_mask_flat = get_fss_mask(weight_flat, local_bank_num, self.current_sparsity)
                new_mask = new_mask_flat.reshape(heads, d_out, d_in)
            else:
                new_mask = get_fss_mask(weight_to_prune, local_bank_num, self.current_sparsity)
            
            mask_buffer.copy_(new_mask)
            with torch.no_grad():
                param.masked_fill_(~mask_buffer, 0.0)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self.target_sparsity <= 0 or self.current_sparsity == 0: return
        
        with torch.no_grad():
            for name, (parent_module, param_attr, mask_name, _, _) in self.pruned_params.items():
                param = getattr(parent_module, param_attr)
                mask_buffer = getattr(parent_module, mask_name)
                if param.grad is not None:
                    param.grad.masked_fill_(~mask_buffer, 0.0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.target_sparsity <= 0 or self.current_sparsity == 0: return
        
        if trainer.global_step > self._last_global_step:
            self._last_global_step = trainer.global_step
            with torch.no_grad():
                for name, (parent_module, param_attr, mask_name, _, _) in self.pruned_params.items():
                    param = getattr(parent_module, param_attr)
                    mask_buffer = getattr(parent_module, mask_name)
                    param.masked_fill_(~mask_buffer, 0.0)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.target_sparsity <= 0 or self.current_sparsity == 0: return

        # 1. Visualization & Reporting
        self._report_and_visualize(pl_module)
        
        # 2. Deep Diagnostics (New)
        self._run_deep_diagnostics(pl_module)

        total_violations = 0
        total_nnz_all = 0
        total_params_all = 0
        
        for name, (parent_module, param_attr, mask_name, is_3d, local_bank_num) in self.pruned_params.items():
            param = getattr(parent_module, param_attr)
            mask_buffer = getattr(parent_module, mask_name)
            
            try:
                # Check Compliance
                if is_3d:
                    heads, d_out, d_in = param.shape
                    weight_flat = param.reshape(-1, d_in)
                    mask_flat = mask_buffer.reshape(-1, d_in)
                    stats = check_fss_compliance(weight_flat, mask_flat, local_bank_num)
                else:
                    stats = check_fss_compliance(param, mask_buffer, local_bank_num)
                    
                total_nnz_all += stats['total_nnz']
                total_params_all += stats['total_params']
            except AssertionError as e:
                print(f"[FSS ERROR] Constraint violation in {name}: {e}")
                total_violations += 1
        
        if total_violations > 0:
            print(f"[FSS CRITICAL] Found {total_violations} parameters violating FSS constraints!")
        
        actual_global_sparsity = 1.0 - (total_nnz_all / total_params_all) if total_params_all > 0 else 0.0
        
        pl_module.log("pruning/target_sparsity", self.current_sparsity, on_epoch=True)
        pl_module.log("pruning/actual_global_sparsity", actual_global_sparsity, on_epoch=True)
        pl_module.log("pruning/violations", float(total_violations), on_epoch=True)

    def _run_deep_diagnostics(self, pl_module):
        """
        Perform strict checks on memory layout, pointer stability, and physical sparsity patterns.
        Crucial for custom CUDA kernels like sLSTM.
        """
        print(f"\n[FSS Deep Diagnostics] Epoch {pl_module.current_epoch}")
        
        for name, (parent, attr, mask_name, is_3d, local_bank_num) in self.pruned_params.items():
            # Skip FFN layers for deep diagnostics to reduce noise, focus on sLSTM
            if "proj_up" in name or "proj_down" in name: continue
            
            param = getattr(parent, attr)
            mask = getattr(parent, mask_name)
            
            print(f"  Target: {name}")
            
            # --- A) Semantic Screening (New) ---
            # Check for high-risk keywords in name or parent class
            risk_keywords = ["ext2int", "int2ext", "packed", "fused", "proxy", "cache", "_internal"]
            parent_type = str(type(parent))
            semantic_risk = False
            if any(k in name for k in risk_keywords) or any(k in parent_type.lower() for k in risk_keywords):
                print(f"    [SEMANTIC RISK] Name or Parent '{parent_type}' suggests internal packing/fused ops.")
                print(f"                    Pruning this might violate internal layout assumptions.")
                semantic_risk = True

            # --- B) Layout Screening ---
            
            # 1. Pointer & Stride Stability
            # Note: We can't compare 'before' and 'after' easily here without hooks, 
            # but we can check if it looks healthy (contiguous).
            print(f"    - Memory: Ptr={param.data_ptr()} | Stride={param.stride()} | Contiguous={param.is_contiguous()}")
            if not param.is_contiguous():
                print(f"    [RISK] Parameter is NOT contiguous! In-place masking might be unsafe or layout is permuted.")

            # 2. Gradient Consistency
            if param.grad is not None:
                grad_zeros = (param.grad == 0).sum().item()
                mask_zeros = (~mask).sum().item()
                # Check intersection: grad is 0 WHERE mask is 0
                overlap = ((param.grad == 0) & (~mask)).sum().item()
                
                print(f"    - Gradients: Masked Elements={mask_zeros} | Zero Grads={grad_zeros} | Overlap={overlap}")
                if overlap != mask_zeros:
                    diff = mask_zeros - overlap
                    print(f"    [CRITICAL WARNING] {diff} elements are Masked but have Non-Zero Gradients! Autograd leakage detected.")
                else:
                    print(f"    [PASS] Gradient masking is consistent.")
            else:
                print(f"    - Gradients: None (Skipped)")

            # 3. Physical 2:4 Layout Check
            # If we are in 2:4 mode (bank_size=4), check if PHYSICAL memory is 2:4 sparse.
            if self.bank_size == 4:
                # View as flat physical memory
                flat_mem = param.view(-1)
                # Check if total size is divisible by 4
                if flat_mem.numel() % 4 == 0:
                    chunks = flat_mem.view(-1, 4)
                    nnz_per_chunk = (chunks != 0).sum(dim=1)
                    # We expect <= 2 non-zeros per chunk for 2:4 sparsity (if target is 50%)
                    # Strictly, FSS enforces exactly K non-zeros in the LOGICAL bank.
                    # If logical bank aligns with physical chunk, this should pass.
                    compliant = (nnz_per_chunk == 2).sum().item()
                    total = chunks.shape[0]
                    ratio = compliant / total
                    print(f"    - Physical 2:4 Alignment: {compliant}/{total} chunks ({ratio:.1%}) match 2:4 pattern.")
                    
                    if ratio < 0.99 and self.current_sparsity >= 0.5:
                        print(f"    [LAYOUT MISMATCH] Logical banking does NOT match physical storage!")
                        if semantic_risk:
                            print(f"    [VERDICT] HIGH RISK. Recommend excluding '{name}' from pruning.")
                        else:
                            print(f"    [VERDICT] Dimension mismatch. Check if pruning_dim ({param.shape[-1]}) is the contiguous dim.")
                else:
                    print(f"    - Physical 2:4: Skipped (numel {flat_mem.numel()} not divisible by 4)")

    def _report_and_visualize(self, pl_module):
        if not self.pruned_params: return
        
        # --- 1. Text Report (Console) - Keep only Representative ---
        # Pick the first pruned parameter as representative for console spam
        rep_name = list(self.pruned_params.keys())[0]
        # ... (Console printing logic for rep_name only) ...
        # (Since user asked to UPLOAD all heatmaps, but console text is too long for all 5, 
        # let's keep console text for 1, but loop for wandb images)
        
        parent, attr, mask_name, is_3d, local_bank_num = self.pruned_params[rep_name]
        weight = getattr(parent, attr)
        mask = getattr(parent, mask_name)
        
        if is_3d: w_sample, m_sample = weight[0], mask[0]
        else: w_sample, m_sample = weight, mask
             
        rows, cols = w_sample.shape
        bank_size = cols // local_bank_num
        m_reshaped = m_sample.reshape(rows, local_bank_num, bank_size)
        nnz_per_bank = m_reshaped.sum(dim=2)
        
        print(f"\n[FSS Report] Inspecting {rep_name} (Representative)")
        print(f"Bank Size: {bank_size} | Expected NNZ per bank: {int(nnz_per_bank[0,0].item())}")
        print("NNZ per bank (first 4 rows):")
        print(nnz_per_bank[:4].int())
        
        r_slice, c_slice = 8, min(32, cols) 
        torch.set_printoptions(precision=4, sci_mode=False, linewidth=160)
        print(f"--- Weight Slice ---")
        print(w_sample[:r_slice, :c_slice])
        print(f"--- Mask Slice ---")
        print(m_sample[:r_slice, :c_slice].int())
        torch.set_printoptions(profile="default")

        # --- 2. WandB Visualization (Heatmap) - ALL Parameters ---
        try:
            import wandb
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            
            if isinstance(pl_module.logger, list): logger = pl_module.logger[0]
            else: logger = pl_module.logger
                
            if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'log'):
                # Loop through ALL pruned parameters
                for name, (parent, attr, mask_name, is_3d, _) in self.pruned_params.items():
                    weight = getattr(parent, attr)
                    if is_3d: w_sample = weight[0] # Take first head for 3D
                    else: w_sample = weight
                    
                    w_np = w_sample.detach().float().cpu().numpy()
                    w_abs = np.abs(w_np)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    im = ax.imshow(w_abs, aspect='auto', cmap='viridis', interpolation='nearest')
                    plt.colorbar(im, ax=ax, label='Weight Magnitude')
                    
                    # Shorten name for title
                    short_name = name.split("xlstm.blocks.")[-1] if "xlstm.blocks." in name else name
                    ax.set_title(f"{short_name}\nSparsity: {self.current_sparsity:.2f}")
                    ax.set_xlabel("In Features")
                    ax.set_ylabel("Out Features")
                    
                    logger.experiment.log({
                        f"pruning/heatmap/{short_name}": wandb.Image(fig),
                        "global_step": pl_module.global_step
                    })
                    plt.close(fig)
        except Exception as e:
            print(f"[FSS Viz Warning] Could not log heatmap to WandB: {e}")
