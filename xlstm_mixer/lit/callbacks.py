from lightning.pytorch.callbacks import Callback
import torch
import torch.nn as nn
from ..utils.pruning_algo import get_fss_mask, check_fss_compliance


def _normalize_str_list(x):
    """
    Accept None / list[str] / tuple[str] / comma-separated string (or bracketed list-ish string)
    and normalize to list[str] or None.
    """
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out or None
    s = str(x).strip()
    if not s:
        return None
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip().strip('"').strip("'") for p in s.split(",")]
    parts = [p for p in parts if p]
    return parts or None


class FSSPruningCallback(Callback):
    def __init__(
        self,
        target_sparsity: float = 0.0,  # Default 0 = disabled
        bank_num: int = 4,
        bank_size: int = None,  # If set, overrides bank_num dynamically per layer
        start_epoch: int = 0,
        warmup_epochs: int = 10,
        pruning_frequency: int = 1,
        # Substring match against parameter names.
        # - None: default to gate weights + recurrent kernel
        # - list/tuple/str: any keyword match enables pruning for that parameter
        target_modules=None,
        # If True (default), only prune params inside xLSTM blocks ("xlstm.blocks" in name).
        # Set False to allow pruning top-level linear layers (mlp_in/pre_encoding/fc/Linear, etc.)
        blocks_only: bool = True,
        # Optional: further restrict to names that contain any of these substrings.
        include_scopes=None,
        # Override default exclude keywords (defaults exclude FFN/proj, bias, norm).
        exclude_keywords=None,
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
        self.bank_size = bank_size  # Store bank_size preference
        self.start_epoch = start_epoch
        self.warmup_epochs = warmup_epochs
        self.pruning_frequency = pruning_frequency

        self.blocks_only = blocks_only
        self.include_scopes = _normalize_str_list(include_scopes)

        target_modules_norm = _normalize_str_list(target_modules)
        if target_modules_norm is None:
            self.target_keywords = [
                "igate",
                "fgate",
                "ogate",
                "zgate",
                "_recurrent_kernel_",
            ]
        else:
            self.target_keywords = target_modules_norm

        exclude_norm = _normalize_str_list(exclude_keywords)
        self.exclude_keywords = (
            exclude_norm
            if exclude_norm is not None
            else ["proj_up", "proj_down", "bias", "norm"]
        )  # Explicitly exclude FFN by default

        self.current_sparsity = 0.0
        # pruned_modules stores: name -> (parent_module, param_name, mask_name, is_3d)
        self.pruned_params = {}
        self._last_global_step = -1
        # 存储稀疏度vs指标的历史数据，用于绘制曲线
        self._sparsity_history = []  # List of (sparsity, metrics_dict)
        # 标记是否已定义WandB指标
        self._wandb_metrics_defined = False

    def _is_target_param(self, name):
        if self.blocks_only and "xlstm.blocks" not in name:
            return False
        if self.include_scopes is not None and not any(
            s in name for s in self.include_scopes
        ):
            return False
        if any(k in name for k in self.exclude_keywords):
            return False
        if not any(k in name for k in self.target_keywords):
            return False
        return True

    def setup(self, trainer, pl_module, stage=None):
        if self.target_sparsity <= 0:
            return

        if stage in (None, "fit"):
            count = 0
            # Scan parameters instead of modules
            for name, param in pl_module.named_parameters():
                if not param.requires_grad:
                    continue

                if self._is_target_param(name):
                    # name is like 'model.xlstm.blocks.0.xlstm.fgate.weight'
                    # parent module name is 'model.xlstm.blocks.0.xlstm.fgate'
                    # param name is 'weight'

                    split_name = name.rsplit(".", 1)
                    if len(split_name) != 2:
                        continue
                    parent_name, param_attr = split_name

                    # Find parent module
                    parent_module = pl_module
                    if parent_name:
                        for part in parent_name.split("."):
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
                            elif part.isdigit():  # Sequential or ModuleList
                                parent_module = parent_module[int(part)]
                            else:
                                # Could not traverse
                                print(
                                    f"[FSS Warning] Could not find parent module for {name}"
                                )
                                parent_module = None
                                break

                    if parent_module is None:
                        continue

                    # Check dimensions
                    # Case 1: 3D [Heads, Out, In] -> [8, 32, 32] or [8, 32, 128]
                    # Case 2: 2D [Out, In] -> standard Linear

                    is_3d = False
                    if param.dim() == 3:
                        is_3d = True
                        heads, d_out, d_in = param.shape
                        pruning_dim = d_in  # Usually the last dim is input features
                    elif param.dim() == 2:
                        d_out, d_in = param.shape
                        pruning_dim = d_in
                    else:
                        continue  # Skip 1D biases or other shapes

                    # Determine Bank Num (Fixed Num vs Fixed Size)
                    if self.bank_size is not None:
                        # Mode 2: Fixed Bank Size (e.g. 4 for 2:4 sparsity)
                        if pruning_dim % self.bank_size != 0:
                            print(
                                f"[FSS Skip] {name}: dim ({pruning_dim}) % bank_size ({self.bank_size}) != 0"
                            )
                            continue
                        local_bank_num = pruning_dim // self.bank_size
                        if local_bank_num < 1:
                            continue
                    else:
                        # Mode 1: Fixed Bank Num (Original FSS)
                        if pruning_dim % self.bank_num != 0:
                            print(
                                f"[FSS Skip] {name}: pruning dim ({pruning_dim}) % {self.bank_num} != 0"
                            )
                            continue
                        local_bank_num = self.bank_num

                    mask_name = f"{param_attr}_mask"

                    # Register Buffer
                    if not hasattr(parent_module, mask_name):
                        mask = torch.ones_like(param, dtype=torch.bool)
                        parent_module.register_buffer(mask_name, mask)

                    # Store local_bank_num for later use
                    self.pruned_params[name] = (
                        parent_module,
                        param_attr,
                        mask_name,
                        is_3d,
                        local_bank_num,
                    )
                    count += 1

                    # Initial Apply
                    with torch.no_grad():
                        mask = getattr(parent_module, mask_name)
                        param.masked_fill_(~mask, 0.0)

            print(
                f"[FSS Setup] Managing FSS masks for {len(self.pruned_params)} parameters. Target Sparsity: {self.target_sparsity}"
            )
            if self.bank_size:
                print(
                    f"[FSS Config] Mode: Fixed Bank Size = {self.bank_size} (Dynamic Bank Num per layer)"
                )
            else:
                print(f"[FSS Config] Mode: Fixed Bank Num = {self.bank_num}")

    def on_train_epoch_start(self, trainer, pl_module):
        if self.target_sparsity <= 0:
            return

        current_epoch = trainer.current_epoch
        if current_epoch < self.start_epoch:
            return

        if current_epoch >= self.start_epoch + self.warmup_epochs:
            self.current_sparsity = self.target_sparsity
        else:
            t = current_epoch - self.start_epoch
            T = self.warmup_epochs
            self.current_sparsity = self.target_sparsity * (1 - (1 - t / T) ** 3)

        if (current_epoch - self.start_epoch) % self.pruning_frequency == 0:
            if self.current_sparsity > 0:
                print(
                    f"\n[FSS Pruning] Epoch {current_epoch}: Updating masks to sparsity {self.current_sparsity:.4f}"
                )
                self._update_masks()

    def _update_masks(self):
        for name, (
            parent_module,
            param_attr,
            mask_name,
            is_3d,
            local_bank_num,
        ) in self.pruned_params.items():
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
        if self.target_sparsity <= 0 or self.current_sparsity == 0:
            return

        with torch.no_grad():
            for name, (parent_module, param_attr, mask_name, _, _) in self.pruned_params.items():
                param = getattr(parent_module, param_attr)
                mask_buffer = getattr(parent_module, mask_name)
                if param.grad is not None:
                    param.grad.masked_fill_(~mask_buffer, 0.0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.target_sparsity <= 0 or self.current_sparsity == 0:
            return

        if trainer.global_step > self._last_global_step:
            self._last_global_step = trainer.global_step
            with torch.no_grad():
                for name, (parent_module, param_attr, mask_name, _, _) in self.pruned_params.items():
                    param = getattr(parent_module, param_attr)
                    mask_buffer = getattr(parent_module, mask_name)
                    param.masked_fill_(~mask_buffer, 0.0)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.target_sparsity <= 0 or self.current_sparsity == 0:
            return

        # 1. Visualization & Reporting
        self._report_and_visualize(pl_module)

        # 2. Deep Diagnostics (New)
        self._run_deep_diagnostics(pl_module)

        # ====== A. 对 FSS 目标层做合规检查（沿用原逻辑） ======
        total_violations = 0
        total_nnz_fss = 0
        total_params_fss = 0

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

                total_nnz_fss += stats["total_nnz"]
                total_params_fss += stats["total_params"]
            except AssertionError as e:
                print(f"[FSS ERROR] Constraint violation in {name}: {e}")
                total_violations += 1

        if total_violations > 0:
            print(f"[FSS CRITICAL] Found {total_violations} parameters violating FSS constraints!")

        # ====== B. 计算“全模型维度”的 global sparsity ======
        global_nnz = 0
        global_params = 0

        for name, param in pl_module.named_parameters():
            # 只统计需要梯度的权重矩阵，跳过 bias / 标量等
            if not param.requires_grad:
                continue
            if param.dim() < 2:
                continue

            numel = param.numel()
            nnz = (param != 0).sum().item()
            global_nnz += nnz
            global_params += numel

        if global_params > 0:
            actual_global_sparsity = 1.0 - (global_nnz / global_params)
        else:
            actual_global_sparsity = 0.0

        pl_module.log("pruning/target_sparsity", self.current_sparsity, on_epoch=True)
        pl_module.log("pruning/actual_global_sparsity", actual_global_sparsity, on_epoch=True)
        pl_module.log("pruning/violations", float(total_violations), on_epoch=True)

        # 3. Log sparsity vs accuracy curves（使用新的 global sparsity）
        self._log_sparsity_vs_metrics(pl_module, actual_global_sparsity)

        # 4. Log detailed comparison metrics
        self._log_detailed_comparison_metrics(pl_module)

    def _run_deep_diagnostics(self, pl_module):
        """
        Perform strict checks on memory layout, pointer stability, and physical sparsity patterns.
        Crucial for custom CUDA kernels like sLSTM.
        """
        print(f"\n[FSS Deep Diagnostics] Epoch {pl_module.current_epoch}")

        for name, (parent, attr, mask_name, is_3d, local_bank_num) in self.pruned_params.items():
            # Skip FFN layers for deep diagnostics to reduce noise, focus on sLSTM
            if "proj_up" in name or "proj_down" in name:
                continue

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
            print(
                f"    - Memory: Ptr={param.data_ptr()} | Stride={param.stride()} | Contiguous={param.is_contiguous()}"
            )
            if not param.is_contiguous():
                print(
                    f"    [RISK] Parameter is NOT contiguous! In-place masking might be unsafe or layout is permuted."
                )

            # 2. Gradient Consistency
            if param.grad is not None:
                grad_zeros = (param.grad == 0).sum().item()
                mask_zeros = (~mask).sum().item()
                # Check intersection: grad is 0 WHERE mask is 0
                overlap = ((param.grad == 0) & (~mask)).sum().item()

                print(
                    f"    - Gradients: Masked Elements={mask_zeros} | Zero Grads={grad_zeros} | Overlap={overlap}"
                )
                if overlap != mask_zeros:
                    diff = mask_zeros - overlap
                    print(
                        f"    [CRITICAL WARNING] {diff} elements are Masked but have Non-Zero Gradients! Autograd leakage detected."
                    )
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
                    print(
                        f"    - Physical 2:4 Alignment: {compliant}/{total} chunks ({ratio:.1%}) match 2:4 pattern."
                    )

                    if ratio < 0.99 and self.current_sparsity >= 0.5:
                        print(f"    [LAYOUT MISMATCH] Logical banking does NOT match physical storage!")
                        if semantic_risk:
                            print(f"    [VERDICT] HIGH RISK. Recommend excluding '{name}' from pruning.")
                        else:
                            print(
                                f"    [VERDICT] Dimension mismatch. Check if pruning_dim ({param.shape[-1]}) is the contiguous dim."
                            )
                else:
                    print(
                        f"    - Physical 2:4: Skipped (numel {flat_mem.numel()} not divisible by 4)"
                    )

    def _report_and_visualize(self, pl_module):
        if not self.pruned_params:
            return

        # --- 1. Text Report (Console) - Keep only Representative ---
        # Pick the first pruned parameter as representative for console spam
        rep_name = list(self.pruned_params.keys())[0]
        # ... (Console printing logic for rep_name only) ...
        # (Since user asked to UPLOAD all heatmaps, but console text is too long for all 5,
        # let's keep console text for 1, but loop for wandb images)

        parent, attr, mask_name, is_3d, local_bank_num = self.pruned_params[rep_name]
        weight = getattr(parent, attr)
        mask = getattr(parent, mask_name)

        if is_3d:
            w_sample, m_sample = weight[0], mask[0]
        else:
            w_sample, m_sample = weight, mask

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

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            if isinstance(pl_module.logger, list):
                logger = pl_module.logger[0]
            else:
                logger = pl_module.logger

            if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
                # Loop through ALL pruned parameters
                for name, (parent, attr, mask_name, is_3d, _) in self.pruned_params.items():
                    weight = getattr(parent, attr)
                    if is_3d:
                        w_sample = weight[0]  # Take first head for 3D
                    else:
                        w_sample = weight

                    w_np = w_sample.detach().float().cpu().numpy()
                    w_abs = np.abs(w_np)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im = ax.imshow(w_abs, aspect="auto", cmap="gray_r", interpolation="nearest")
                    plt.colorbar(im, ax=ax, label="Weight Magnitude")

                    # Shorten name for title
                    short_name = name.split("xlstm.blocks.")[-1] if "xlstm.blocks." in name else name
                    ax.set_title(f"{short_name}\nSparsity: {self.current_sparsity:.2f}")
                    ax.set_xlabel("In Features")
                    ax.set_ylabel("Out Features")

                    logger.experiment.log(
                        {f"pruning/heatmap/{short_name}": wandb.Image(fig), "global_step": pl_module.global_step}
                    )
                    plt.close(fig)
        except Exception as e:
            print(f"[FSS Viz Warning] Could not log heatmap to WandB: {e}")

    def _log_sparsity_vs_metrics(self, pl_module, actual_sparsity):
        """
        记录稀疏度与验证指标的关系，用于绘制稀疏度vs精度曲线

        记录原始数据点，可在WandB界面中手动创建自定义图表：
        - pruning/sparsity: 稀疏度值
        - pruning/loss_at_sparsity: 对应稀疏度下的loss
        - pruning/mse_at_sparsity: 对应稀疏度下的MSE
        - pruning/mae_at_sparsity: 对应稀疏度下的MAE
        - 等等
        """
        try:
            import wandb

            # 获取logger
            if isinstance(pl_module.logger, list):
                logger = pl_module.logger[0]
            else:
                logger = pl_module.logger

            if not hasattr(logger, "experiment") or not hasattr(logger.experiment, "log"):
                return

            # 定义WandB指标，指定以pruning/sparsity作为x轴
            if not self._wandb_metrics_defined and hasattr(logger.experiment, "define_metric"):
                metrics_to_define = ["loss", "mse", "mae", "mape", "smape"]
                for metric_name in metrics_to_define:
                    logger.experiment.define_metric(
                        f"pruning/{metric_name}_at_sparsity", step_metric="pruning/sparsity"
                    )
                self._wandb_metrics_defined = True

            # 获取当前epoch的验证指标
            logged_metrics = {}
            if hasattr(pl_module.trainer, "logged_metrics"):
                logged_metrics = pl_module.trainer.logged_metrics

            # 尝试获取常见的验证指标
            metrics_to_track = {
                "val/loss": "loss",
                "val/MeanSquaredError": "mse",
                "val/MeanAbsoluteError": "mae",
                "val/MeanAbsolutePercentageError": "mape",
                "val/SymmetricMeanAbsolutePercentageError": "smape",
            }

            # 收集当前epoch的指标值
            current_metrics = {}
            for metric_key, metric_name in metrics_to_track.items():
                if metric_key in logged_metrics:
                    metric_value = logged_metrics[metric_key]
                    # 如果是tensor，转换为float
                    if hasattr(metric_value, "item"):
                        metric_value = metric_value.item()
                    elif hasattr(metric_value, "cpu"):
                        metric_value = metric_value.cpu().item()
                    current_metrics[metric_name] = metric_value

            # 存储历史数据
            self._sparsity_history.append((actual_sparsity, current_metrics.copy()))

            # 方法1：记录原始数据点（方便在WandB中创建自定义图表）
            log_dict = {}
            for metric_name, metric_value in current_metrics.items():
                log_dict[f"pruning/{metric_name}_at_sparsity"] = metric_value

            if log_dict:
                log_dict["pruning/sparsity"] = actual_sparsity
                log_dict["global_step"] = pl_module.global_step
                logger.experiment.log(log_dict, commit=False)

            # 提交所有记录
            logger.experiment.log({}, commit=True)

        except Exception as e:
            print(f"[FSS Warning] Could not log sparsity vs metrics: {e}")

    def _log_detailed_comparison_metrics(self, pl_module):
        """
        记录详细的对比指标，用于多维度分析：
        - bank划分一致性
        - keep_k变化
        - bank内weight rank distribution
        - 非零分布偏差
        - 硬件负载平衡
        - 剪枝稳定性
        """
        if self.target_sparsity <= 0 or self.current_sparsity == 0:
            return

        try:
            import wandb
            import numpy as np

            # 获取logger
            if isinstance(pl_module.logger, list):
                logger = pl_module.logger[0]
            else:
                logger = pl_module.logger

            if not hasattr(logger, "experiment") or not hasattr(logger.experiment, "log"):
                return

            # 汇总统计
            all_bank_sizes = []
            all_bank_nums = []
            all_keep_ks = []
            all_nnz_variances = []
            all_load_imbalances = []
            all_mask_stabilities = []

            # 对每个参数进行详细分析
            for name, (parent_module, param_attr, mask_name, is_3d, local_bank_num) in self.pruned_params.items():
                param = getattr(parent_module, param_attr)
                mask_buffer = getattr(parent_module, mask_name)

                if is_3d:
                    heads, d_out, d_in = param.shape
                    weight_flat = param.reshape(-1, d_in)
                    mask_flat = mask_buffer.reshape(-1, d_in)
                else:
                    weight_flat = param
                    mask_flat = mask_buffer

                rows, cols = weight_flat.shape
                bank_size = cols // local_bank_num
                keep_k = int(round(bank_size * (1 - self.current_sparsity)))
                keep_k = max(1, min(keep_k, bank_size))

                # 1. Bank划分一致性统计
                all_bank_sizes.append(bank_size)
                all_bank_nums.append(local_bank_num)
                all_keep_ks.append(keep_k)

                # 2. Bank内非零分布偏差（variance）
                mask_reshaped = mask_flat.reshape(rows, local_bank_num, bank_size)
                nnz_per_bank = mask_reshaped.sum(dim=2).float()  # [rows, bank_num]

                # 计算每行的variance（bank间的一致性）
                nnz_variance = nnz_per_bank.var(dim=1).mean().item()  # 平均variance
                all_nnz_variances.append(nnz_variance)

                # 3. Bank内weight rank distribution
                weight_abs = torch.abs(weight_flat)
                weight_reshaped = weight_abs.reshape(rows, local_bank_num, bank_size)

                # 对每个bank内的权重进行排序，计算rank分布
                # 这里计算保留的权重在bank内的平均rank位置
                ranks = []
                for r in range(min(10, rows)):  # 采样前10行
                    for b in range(local_bank_num):
                        bank_weights = weight_reshaped[r, b, :]
                        bank_mask = mask_reshaped[r, b, :]
                        if bank_mask.sum() > 0:
                            # 计算保留权重的rank位置（归一化到0-1）
                            sorted_indices = torch.argsort(bank_weights, descending=True)
                            kept_indices = sorted_indices[:keep_k]
                            # rank位置 = 在bank_size中的位置 / bank_size
                            rank_positions = (kept_indices.float() / bank_size).mean().item()
                            ranks.append(rank_positions)

                # 4. 硬件负载平衡（bank内外load imbalance）
                # 计算bank内和bank间的权重分布不均匀度
                bank_weights_sum = weight_reshaped.sum(dim=2)  # [rows, bank_num]
                row_weights_sum = bank_weights_sum.sum(dim=1)  # [rows]

                # Bank间imbalance: 同一行不同bank的权重和差异
                bank_imbalance = (
                    (bank_weights_sum.std(dim=1) / (bank_weights_sum.mean(dim=1) + 1e-8)).mean().item()
                )

                # Bank内imbalance: 同一bank内不同位置的权重差异
                bank_internal_imbalance = (
                    (weight_reshaped.std(dim=2) / (weight_reshaped.mean(dim=2) + 1e-8)).mean().item()
                )

                total_imbalance = bank_imbalance + bank_internal_imbalance
                all_load_imbalances.append(total_imbalance)

                # 5. 剪枝稳定性（mask变化率）
                # 如果有历史mask，计算变化率
                if hasattr(self, "_previous_masks") and name in self._previous_masks:
                    prev_mask = self._previous_masks[name]
                    if prev_mask.shape == mask_flat.shape:
                        mask_change = (mask_flat != prev_mask).float().mean().item()
                        all_mask_stabilities.append(1.0 - mask_change)  # 稳定性 = 1 - 变化率

                # 存储当前mask用于下次比较
                if not hasattr(self, "_previous_masks"):
                    self._previous_masks = {}
                self._previous_masks[name] = mask_flat.clone()

            # 记录汇总统计
            if all_bank_sizes:
                # Bank划分一致性
                pl_module.log("comparison/avg_bank_size", np.mean(all_bank_sizes), on_epoch=True)
                pl_module.log("comparison/avg_bank_num", np.mean(all_bank_nums), on_epoch=True)
                pl_module.log("comparison/bank_size_std", np.std(all_bank_sizes), on_epoch=True)
                pl_module.log("comparison/bank_num_std", np.std(all_bank_nums), on_epoch=True)

                # Keep_k统计
                pl_module.log("comparison/avg_keep_k", np.mean(all_keep_ks), on_epoch=True)
                pl_module.log("comparison/keep_k_std", np.std(all_keep_ks), on_epoch=True)
                pl_module.log("comparison/keep_k_min", np.min(all_keep_ks), on_epoch=True)
                pl_module.log("comparison/keep_k_max", np.max(all_keep_ks), on_epoch=True)

                # 非零分布偏差
                if all_nnz_variances:
                    pl_module.log("comparison/avg_nnz_variance", np.mean(all_nnz_variances), on_epoch=True)
                    pl_module.log("comparison/max_nnz_variance", np.max(all_nnz_variances), on_epoch=True)

                # 硬件负载平衡
                if all_load_imbalances:
                    pl_module.log("comparison/avg_load_imbalance", np.mean(all_load_imbalances), on_epoch=True)
                    pl_module.log("comparison/max_load_imbalance", np.max(all_load_imbalances), on_epoch=True)

                # 剪枝稳定性
                if all_mask_stabilities:
                    pl_module.log("comparison/avg_mask_stability", np.mean(all_mask_stabilities), on_epoch=True)
                    pl_module.log("comparison/min_mask_stability", np.min(all_mask_stabilities), on_epoch=True)

            # 记录配置信息（用于对比）
            pl_module.log("comparison/bank_size_config", self.bank_size if self.bank_size else self.bank_num, on_epoch=True)
            pl_module.log("comparison/target_sparsity", self.target_sparsity, on_epoch=True)
            pl_module.log("comparison/current_sparsity", self.current_sparsity, on_epoch=True)

        except Exception as e:
            print(f"[FSS Warning] Could not log detailed comparison metrics: {e}")

