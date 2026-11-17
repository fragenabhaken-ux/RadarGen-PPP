# Architecture brief for the four implemented radar-prediction ablations

Prepared 2026-11-13 as input for the paper-writing agent. This describes the implemented production models, not the larger list of proposed ablations. Architecture and initialization statements below are grounded in local source and production configs. Motivations are experimental hypotheses, not conclusions established by the results. This is an implementation brief; external architectural citations should be obtained separately when drafting the paper.

## 1. Scope and terminology

The four variants are:

1. **Random-initialized DINOv3 architecture with the full transformer neck and standard head** (`ablation_dino_random_init`).
2. **ImageNet-pretrained ResNet-50 with the same full transformer neck and standard head** (`ablation_resnet50_backbone`).
3. **DiT with the original trainable DC-AE output decoder, adapted to remove the 32-channel output bottleneck** (`ablation_dit_dcae_no_bottleneck`).
4. **Pretrained DINOv3 with a compact internal transformer-neck parameterization and the standard head** (`ablation_dino_neck_compact`).

“Backbone”, “neck” and “head” are different components. In the direct DINO/ResNet predictors, the backbone independently encodes each conditioning map; the neck fuses and refines the concatenated modality features; the head reconstructs dense radar fields. The **12 transformer blocks of the DINO backbone are additional to the 12 transformer blocks in the neck**. ResNet replaces the backbone only: it does not remove the transformer neck. Compact DINO also retains all 12 neck blocks.

All four are feed-forward predictors of marked Poisson point process (PPP) fields. Although the DiT inherits a diffusion-model architecture and pretrained assets, its PPP forward pass does **not** denoise a noisy radar latent, iterate diffusion steps, or train a diffusion noise-prediction objective.

## 2. Shared task, conditioning and outputs

### Inputs

Each sample provides three spatially aligned, three-channel bird’s-eye-view conditioning maps, in this order:

- Appearance: `bev_color_map`.
- Semantics: `bev_seg_map`.
- Velocity: `bev_velocity_map`.

Each has shape **B × 3 × 512 × 512**. They are preprocessed BEV representations derived from the camera inputs, not raw perspective camera images. The production setup uses the four configured TruckScenes camera views, and a BEV coordinate range of **±50 m**. The same preprocessing signature and dataset inventory are used across the four ablations. Existing conditioning-map normalization to approximately **[-1,1]** is retained. In particular, the ResNet substitution does not introduce ImageNet classification normalization/crops, and DINO does not receive an additional natural-image preprocessing pipeline inside the predictor.

The native RGB input layers of the visual backbones are preserved. Appearance, semantics and velocity are not concatenated into a nine-channel image at the backbone input. Instead, they are processed by a shared three-channel encoder and fused afterward. The DiT path encodes these maps into DC-AE latents first; see Section 7.

### Dense radar representation and continuous supervision

All variants predict three raw scalar fields: **log intensity, raw RCS and raw signed Doppler**. For the direct predictors the tensor is B × 3 × Hout × Wout. The DiT wrapper uses an equivalent B × 3 × 1 × Hout × Wout layout, with the extra singleton channel dimension.

The first field is exponentiated and divided by the output-grid area to obtain expected points per cell. Its sum is the expected total count. The RCS and Doppler outputs use the **cubic mark link**: physical mark = raw prediction cubed. The decoder itself outputs unconstrained values; it does not apply sigmoid/tanh bounds to the marks. The configured physical preprocessing ranges are RCS [-20,66] and Doppler [-120,120]; these are not hard decoder-output clamps.

Training uses the **continuous-detection PPP objective**, not the earlier binary occupancy/raster-target formulation. Ground truth consists of individual (x,y,RCS,Doppler) detections. Their coordinates select cells in a piecewise-constant predicted field. Multiple detections falling into the same cell remain multiple observations: they are not deduplicated or averaged into one mark target. Thus “continuous targets” does not imply a continuous-coordinate MLP, bilinear field interpolation, or a decoder emitting an explicit list of points.

For a frame with N detections and M output cells, the implemented intensity term is proportional to:

`[sum_j exp(s_j)/M - sum_i (s_cell(i) - log(M))] / max(N,1)`.

RCS and signed-Doppler L1 errors are evaluated at individual detections, normalized per frame by max(N,1), then averaged over frames. The three terms have weights **1:1:1**. Loss arithmetic uses FP32. Empty frames remain part of the objective. Output-grid normalization follows the actual prediction resolution, so switching from 512 to 1024 does not simply quadruple the expected count at identical log intensity.

Dense predictions are deterministic in evaluation mode for fixed inputs/weights. Point clouds are subsequently sampled from the predicted PPP; sampled-cloud metrics therefore also depend on the evaluation sampling seed. The mark fields are read at sampled point locations. This sampling is distinct from diffusion sampling.

## 3. Standard direct-predictor scaffold

This scaffold is shared by random DINO, compact DINO and ResNet, with the differences described later.

### DINOv3 visual encoder and modality fusion

The DINO-based variants use **`vit_small_patch16_dinov3`**, i.e. a ViT-S/16 architecture with **384-dimensional backbone features, 12 backbone blocks and 6 backbone attention heads**. The installed implementation retains native DINOv3 rotary positional embeddings, four register tokens and the native RGB patch projection; classification/global-pooled representations are not the radar readout. The model extracts normalized **spatial features from the last backbone block only** (`out_indices=[11]`, `feature_fusion=last_concat`). Prefix/register tokens do not become BEV pixels.

The native timm backbone definition uses rotary positional embedding temperature 100, no absolute positional embedding, QKV bias disabled, LayerNorm epsilon 1e-5 and LayerScale initialization 1e-5. These native-backbone details are separate from the added neck's 2D RoPE base, normalization and parallel-block construction described below.

The three modality maps are stacked along the batch dimension to form (3B,3,512,512), passed through the shared backbone, then regrouped. Each yields a **384 × 32 × 32** feature grid. Channel concatenation in appearance/semantics/velocity order gives **B × 1152 × 32 × 32**. There is no learned 1152→384 compression or multilayer feature-fusion module in these ablations. The native 16×16 patch stride is preserved. For nondivisible input sizes, the implementation pads bottom/right and crops the corresponding output; the configured 512 inputs already divide exactly.

### Modality-specific LoRA with a trainable shared backbone

Each DINO backbone contains three modality-specific low-rank adapter sets, one per input type. LoRA is applied to the attention QKV/output and MLP input/output linear layers (`attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2`) through the last selected block. Rank **r=16**, alpha **32**, dropout **0**, scale alpha/r = **2**. The A matrices use Kaiming initialization and the B matrices start at zero.

The underlying visual backbone is **fully trainable** in both implemented DINO ablations. LoRA is retained to give the three conditioning types separate updates in addition to their shared base weights. Do not describe these as frozen-DINO/LoRA-only models. In random DINO, the base weights are trainable random initialization; in compact DINO they are trainable pretrained initialization.

### Full transformer neck

The full neck consumes the concatenated 1152-channel grid and preserves its channel width and spatial grid. It contains **12 parallel transformer blocks**, each with:

- Bias-free pre-LayerNorm over the 1152-dimensional token.
- Attention width **1152 = 9 heads × 128 channels/head**.
- Per-head RMS normalization of Q and K, with learned scales and epsilon 1e-6.
- **2D rotary position encoding** on Q/K, generated from the actual BEV feature grid; half of each head dimension represents each spatial axis. The neck uses frequency base 2000, independently of the backbone’s native positional embedding.
- Global scaled dot-product self-attention over the **32×32 = 1024** spatial tokens, with zero attention dropout.
- A parallel **SELU** MLP branch with hidden width **4608 = 4×1152**. Attention and MLP share the same normalized input; this is not a conventional sequential attention-residual-then-MLP block.
- A fused bias-free input projection **1152→8064**, producing Q/K/V plus MLP features: 3×1152 + 4608.
- Concatenation of attention output and activated MLP features, followed by a bias-free output projection **5760→1152**, and one residual addition.

A final LayerNorm follows the stack. There is no additional downsampling in the neck. Full-neck parameter count: **191,122,176**.

### Standard attention/PixelShuffle head

The class name is **`AttentionPixelShuffleHead`**. This is the shared head for the three direct-predictor ablations, not the old `ProgressivePixelShuffleHead`, and not the DC-AE output decoder.

From the 1152×32×32 input it uses three learned upsampling stages:

| Stage | Projection before PixelShuffle | Upscale | Result |
|---|---|---:|---|
| 1 | 3×3 conv, 1152→2048 | 4 | 128×128×128 (C×H×W) |
| 2 | 3×3 conv, 128→1024 | 4 | 64×512×512 |
| 3 | 3×3 conv, 64→128 | 2 | 32×1024×1024 |

Each projection has stride one and padding one, overlapping neighboring source features before rearrangement. PixelShuffle is a learned subpixel rearrangement, not interpolation of the final radar map. Each stage contains **two** refinement blocks at its output resolution. Each block includes:

1. A local residual unit with two bias-free 3×3 convolutions, GroupNorm with one group, GELU between convolutions and GELU after the residual addition.
2. Pre-normalized **8×8-window multihead attention**, with **4 heads**.
3. A residual, LayerNorm-preconditioned two-layer GELU MLP with expansion factor **2**.

Within each stage, the first refinement block uses ordinary windows; the second shifts the partition by **4 pixels**. The shift uses masked padding and cropping rather than cyclic wrapping, avoiding artificial attention between opposite BEV edges. Padding keys are masked. The final readout is a **1×1 convolution from 32 channels to the three raw radar fields**.

PixelShuffle projection weights use **ICNR initialization**, with identical initial subpixel phases and zero biases, intended to reduce phase/checkerboard artifacts. Overlapping projections, local convolution and shifted windows provide local mixing across patch/window boundaries. These are design motivations; do not claim they prove rotational smoothness or eliminate jagged artifacts without direct evidence.

The head upsamples by **4×4×2=32**, mapping a stride-16 feature grid to **1024×1024 output** from 512×512 inputs (`output_scale=2`). Its count is **23,615,267 parameters**. It remains unchanged in the compact-neck ablation, so lower neck parameter count does not imply proportional reductions in total runtime or high-resolution activation memory. Legacy config fields such as `local_refine_blocks` are not extra blocks in this selected attention head; its actual refinement is defined above.

## 4. Ablation A: random-initialized DINO architecture

### Question and motivation

Test whether the DINOv3 initialization contributes beyond the encoder architecture and subsequent task training. A randomly initialized visual transformer provides a counterpoint to a pretrained visual representation while retaining the same full neck and output head.

### Exact implementation

Use the complete direct DINO scaffold above: ViT-S/16 architecture, modality-specific LoRA, concatenated final-layer 1152-channel features, **12-block/9-head full neck**, and standard **1024-output** head. Every predictor parameter is trainable.

**No DINO pretrained checkpoint, trained PPP checkpoint, or LR-sweep weights are loaded at the first production launch.** The backbone, neck and head are fresh; LoRA retains its usual A-random/B-zero initialization. “Random DINO” is shorthand for a randomly initialized model using the DINOv3 architecture, not a backbone pretrained on random data and not a pretrained backbone with only its classification layer reset.

Parameter breakdown: backbone plus modality LoRA **25,125,888**; full neck **191,122,176**; head **23,615,267**; total **239,863,331**, all trainable.

### Interpretation limits

Architecture matches the full DINO reference, but the historical reference inherited earlier task-trained PPP backbone/adapters and received additional task training. Its comparison with this fresh random model does not alone isolate the causal benefit of DINO pretraining at equal total task-training compute. A strict pretraining ablation would match all downstream initialization/training histories apart from the original backbone weights. Treat the historical comparison as evidence about the available systems, with this caveat.

## 5. Ablation B: ResNet-50 backbone substitution

### Question and motivation

Assess whether the learned BEV representation requires a transformer backbone, or whether a convolutional encoder with locality, hierarchy and weight sharing is competitive when the downstream fusion/readout is retained. This is a **CNN backbone plus transformer neck**, not a transformer-free network.

### Exact implementation

Use standard **Torchvision ResNet-50 / V1.5 stride placement**, initialized from **ImageNet-1K V2** weights. This is not the locally available ResNet-50-v1c alternative. Its standard RGB stem is retained. Classification weights load strictly before the final fully connected layer and global average pooling are removed from the dense prediction path.

The ResNet bottleneck stage depths are **[3,4,6,3]**. The final backbone downsampling is replaced by dilation (`replace_stride_with_dilation=[False,False,True]`), preserving compatible weight shapes while changing final output stride from 32 to **16**. For a 512 input the dense encoder output is **2048×32×32** per modality.

A shared, bias-free **1×1 convolution, 2048→384**, aligns each modality’s feature width with the DINO interface. It is a linear projection without a separately added activation/normalization layer. The same ResNet and projection process all three modalities, stacked as a 3B batch; their projected features are then concatenated to **1152×32×32**.

Use the unchanged **12-block, 1152-wide, 9×128-head full transformer neck** and unchanged standard **1024-output head**. There are **no LoRA adapters** in ResNet. All convolutional weights and BatchNorm affine parameters are trainable; BatchNorm running statistics update normally per worker on the modality-stacked batch. There is no SyncBatchNorm conversion. The backbone is not recomputed under activation checkpointing, avoiding repeated BN running-stat updates; neck/head checkpointing is retained.

Only the ResNet encoder receives ImageNet weights. The 2048→384 projection, neck and head start fresh, with no trained PPP or LR-sweep transfer. The same BEV normalization as DINO is used, rather than adding ImageNet-specific preprocessing.

| Component | Parameters |
|---|---:|
| ResNet-50 encoder, classification layer excluded | 23,508,032 |
| Shared 2048→384 projection | 786,432 |
| Full transformer neck | 191,122,176 |
| Standard head | 23,615,267 |
| **Total, all trainable** | **239,031,907** |

This is only about **0.35% fewer total parameters** than full DINO: the large neck is held fixed. The comparison tests the selected **pretrained backbone systems**, not architecture under identical pretraining: ImageNet supervision, DINO self-supervision, LoRA presence and the required CNN projection differ.

## 6. Ablation C: compact internal DINO neck

### Question and motivation

Test whether the very large full neck is overparameterized and whether its attention/MLP internals can be reduced while preserving the external 1152-channel interface, spatial resolution, depth and standard high-resolution head. This targets parameter efficiency without returning to the earlier externally 384-wide bottleneck design.

### Exact implementation

The encoder uses the original **RGB-pretrained DINOv3 ViT-S/16 checkpoint**. The backbone is fully trainable; LoRA rank 16/alpha 32 remains enabled. LoRA, compact neck and output head start fresh: no trained PPP checkpoint is transferred.

The neck still contains **12 blocks**, takes **1152×32×32** features and returns the same shape. The LayerNorm, per-head Q/K RMS normalization, 2D RoPE, parallel attention/SELU-MLP structure, residual addition and final neck normalization remain unchanged. Only internal branch widths and their corresponding projections change:

| Quantity | Full neck | Compact neck |
|---|---:|---:|
| Residual/token width | 1152 | 1152 |
| Blocks | 12 | 12 |
| Attention heads | 9 | 4 |
| Channels per head | 128 | 128 |
| Total attention width | 1152 | 512 |
| Parallel MLP hidden width | 4608 | 128 |
| Fused QKV+MLP projection | 1152→8064 | **1152→1664** |
| Attention+MLP output projection | 5760→1152 | **640→1152** |

Specifically, the compact fused projection produces **3×(4×128)+128=1664** outputs, split into 512-dimensional Q, K and V plus one 128-dimensional MLP branch. The attention output (512) and SELU-activated MLP branch (128) are concatenated to 640 and projected back to 1152 before residual addition.

The implemented MLP hidden width is **128**, not 256, 1152 or `1152×1`. It is specified explicitly as `neck_mlp_hidden_dim=128`; `neck_head_dim=128` and `neck_heads=4`. Calling it simply “MLP ratio 1” would be misleading without defining the reference width. The standard residual-width-relative ratio would be 128/1152=1/9.

The standard head and 1024 output are unchanged. No 6-block/4-block neck, 384-dimensional residual stream or convolution-only neck is part of this implemented variant.

| Component | Parameters |
|---|---:|
| DINO backbone plus modality LoRA | 25,125,888 |
| Compact transformer neck | 31,869,696 |
| Standard head | 23,615,267 |
| **Total, all trainable** | **80,610,851** |

The neck alone loses **83.3%** of its parameters; the whole predictor loses **66.4%** relative to full DINO. The retained high-resolution head means these savings are not equivalent to a 66.4% latency or memory reduction. This also is not perfectly isolated against historical full-neck checkpoints, because their PPP initialization histories and the chosen training LR differ.

## 7. Ablation D: DiT with original DC-AE decoder without the 32-channel output bottleneck

### Question and motivation

The original PPP adaptation forced 1152-dimensional DiT features through a **32-channel output interface** before decoding, inherited from the autoencoder latent space. This could discard task-relevant information. The later standard-head DiT removed that interface but also changed the decoder design. This ablation tests whether the familiar pretrained DC-AE decoder can be retained while feeding it much wider transformer features, offering an alternative to replacing it with the new shared head.

This hypothesis does not establish that the old bottleneck caused weak results, nor that removing it alone accounts for any measured improvement. Initialization histories and other design differences matter.

### Conditioning and DiT trunk

Model factory **`RadarGenPPP_600M_P1_D28`**: **28 transformer blocks**, hidden width **1152**, patch size **1** on the latent grid. It retains the RadarGen/SANA joint-output-stream structure, linear self-attention, text cross-attention and GLUMBConv feed-forward modules (configured MLP ratio **2.5**, SiLU/SiLU/no final activation), spatial positional embedding and configured Q/K normalization. Factory `num_heads=16` configures the relevant multihead components; the active linear self-attention uses the implementation’s separate `linear_head_dim=32` (1152/32 = 36 heads). Do not equate all DiT attention modules with a 16-head vanilla-attention layer.

The **frozen RGB DC-AE conditioning encoder** maps each 512×512 condition to **32×16×16**, using downsampling factor 32 and scaling factor **0.41407**. The three latents concatenate to 96 channels. For each of three output streams, a learned **three-channel modality embedding** is appended, giving **99 channels**. These identifiers distinguish output fields (density/intensity, RCS and Doppler), not three different raw camera inputs.

The adapted 1×1 DiT input projection maps **99→1152**. Its first 96 channels are initialized by repeating the pretrained 32-channel input weights three times and dividing by three; the three added modality channels retain their fresh initialization. The output streams interact through joint self-attention over their concatenated token sequences; the output decoder itself is shared across streams.

The timestep is fixed to **zero**, and frozen **Gemma-2-2B-it** embeddings of the empty prompt are retained for the text-conditioning branch. There is no noisy-radar input, no predicted diffusion variance and no iterative denoising during PPP inference. The architecture still includes inherited timestep/text-conditioning machinery.

### Exact decoder modification

Original PPP output path:

`(3B,1152,16,16) → DiT final prediction layer → (3B,32,16,16) → shared scalar DC-AE decoder → (3B,1,512,512)`.

Implemented path:

`(3B,1152,16,16) → new 3×3 conv 1152→1024 → existing DC-AE decoder stages → (3B,1,512,512)`.

The DiT final prediction layer is removed/replaced by Identity and the trunk returns full-width spatial features. There is no replacement 1152→32 adapter. The original DC-AE input block combined a **32→1024 3×3 convolution** with a fixed channel-duplication shortcut. That shortcut cannot accept a 1152-channel input. The input block is therefore rebuilt with a **fresh 1152→1024 convolution, stride 1, padding 1, and no input shortcut**. This is a necessary decoder-interface change, so “we only deleted a bottleneck and changed nothing else” is not literally accurate.

The remaining DC-AE decoder topology and pretrained weights are retained. The `dc-ae-f32c32-sana-1.1` decoder has six configured stage widths, listed fine-to-coarse, **[128,256,512,512,1024,1024]**, with **three main blocks per stage**. The first three fine stages use ResBlocks; the three coarse stages use EfficientViT `EViTS5_GLU` blocks. Decoding runs coarse-to-fine, with five 2× `InterpolateConv` upsampling transitions, RMS2D normalization and SiLU activation, giving overall ×32 spatial reconstruction. Internal decoder residual/upsampling shortcuts remain; only the incompatible input shortcut is removed.

The original RGB output convolution is adapted to **one scalar channel**, averaging its three RGB output weight slices and biases. The decoder is shared across the three output streams; there are not three independently parameterized DC-AE decoders. Stream results are reshaped into the three raw radar fields. Full-width features bypass inverse VAE latent scaling, because they are transformer features rather than 32-channel VAE latents. Conditioning encoding still uses its original latent scaling.

Output resolution is **512×512**, unlike the direct DINO/ResNet variants’ 1024×1024. “Without bottleneck” refers to the **output** 32-channel interface; the frozen conditioning encoder still uses 32-channel latents. There is no additional DINO-style 12-block neck in this model.

### Initialization and trainability

The first production allocation uses the same **starting pretrained assets** as historical PPP run **2438928**: original **SANA-600M-512** and RGB **DC-AE f32c32 SANA-1.1**. It does **not** load the trained final checkpoint of 2438928, the later 2516862 checkpoint, a RadarGen downstream checkpoint, or any LR-sweep weights.

The retained DiT trunk and DC-AE output decoder are trainable. The new decoder input convolution, modality embeddings and any checkpoint-missing validated Q/K normalization parameters use fresh initialization. Frozen conditioning DC-AE encoder and Gemma remain outside the trainable prediction-network parameter count.

| Component | Trainable parameters |
|---|---:|
| Adapted DiT trunk, final output layer removed | 591,850,377 |
| Shared scalar DC-AE decoder with widened input | 169,542,785 |
| **Prediction network total** | **761,393,162** |

The original scalar decoder had 159,220,865 parameters; widening its input adds **10,321,920**. These counts exclude the frozen conditioning encoder and text model, but those components still consume memory and execution time. Comparison with the standard-head DiT therefore changes decoder structure and initialization history, and comparison with DINO also changes resolution, conditioning encoder, transformer architecture and optimizer.

## 8. Actual training protocol and run identifiers

Each model was tested with short real-data entry-node updates and a four-GPU batch confirmation, then a fresh approximately 30-minute exponential LR-range test from 1e-7 to 1e-2, without warm-up. LR-sweep weights were discarded. Production models were initialized independently as specified above.

All first production allocations use **four nodes × four GH200 GPUs = 16 GPUs**, bf16 mixed precision, gradient accumulation **1**, seed **42**, deterministic global sample reshuffling with seed+epoch before rank sharding, and activation checkpointing where supported. Training allocation is 12 hours, with an **11.8-hour training guard** to permit orderly saving. Checkpoints are written every **1,000 global optimizer updates**, retained without pruning, plus a final checkpoint on an orderly stop. First allocations use **200 optimizer-update warm-up**, then the selected **constant LR**, with automatic batch-based LR scaling disabled.

| Variant | First production job | Final first-allocation step | Batch/GPU | Global batch | Constant LR | Optimizer |
|---|---:|---:|---:|---:|---:|---|
| Random DINO | 2847666 | 15,000 | 16 | 256 | 1e-4 | AdamW |
| ResNet-50 | 2856174 | 12,575 | 20 | 320 | 1e-4 | AdamW |
| DiT/DC-AE without output bottleneck | 2849053 | 28,537 | 4 | 64 | 1e-4 | CAMEWrapper |
| Compact DINO | 2849062 | 10,372 | 24 | 384 | 1e-5 | AdamW |

AdamW settings: betas **(0.9,0.999)**, epsilon **1e-8**, weight decay **0.01**, gradient clipping **1.0**. DiT CAMEWrapper settings: betas **(0.9,0.999,0.9999)**, epsilons **(1e-30,1e-16)**, weight decay **0**, gradient clipping **0.1**. Task loss weights and target representation remain common.

These are equal allocation sizes, not equal optimizer-update counts, global batch sizes, processed-sample counts or cumulative pretraining compute. Do not describe the experiments as perfectly compute- or data-exposure-matched.

Second allocations were submitted as **exact continuations**: random DINO **2863217**, DiT **2867524**, compact DINO **2867528**, ResNet **2882612**. They load model, optimizer, scheduler, progress and rank RNG states, retain the same LR and batch size, and **do not repeat warm-up**. Original config warm-up values remain for resume-contract compatibility; restored scheduler progress is already beyond them. Submission is not evidence of completion; consult scheduler/checkpoints for current final steps. This brief’s listed first-run evaluations do not incorporate unfinished/later continuations.

ResNet’s earlier launch **2848698** failed before any training update due to the startup prediction diagnostic incorrectly treating ResNet as a latent-model path. The diagnostic was fixed and regression-tested; successful training job **2856174** started fresh. That failed launch is not a separate scientific variant or a trained ancestor.

## 9. Parameter scope and experimental claims

| Predictor | Total trainable prediction parameters |
|---|---:|
| Random DINO / full DINO architecture | 239,863,331 |
| ResNet-50 + full neck/head | 239,031,907 |
| Compact DINO | 80,610,851 |
| DiT with full-width DC-AE output decoder | 761,393,162 |

Counts include the full prediction network, including trainable backbone and adapters, but exclude external BEV preprocessing; DiT counts also exclude its frozen conditioning encoder and Gemma. The original RadarGen reference prediction-network scope is **751,085,100** parameters. Compact DINO has approximately **89.3% fewer** parameters than that scope; the new DiT is slightly larger than the original RadarGen predictor despite removing the narrow interface. Count reduction and bottleneck removal are different concepts.

Recommended claim wording:

- Random initialization tests the value of pretrained visual features within the DINO architecture; current historical comparisons also differ in previous PPP training exposure.
- ResNet tests a convolutional pretrained encoder while retaining the large downstream transformer neck and identical output head; it is not a CNN-only system or a pure equal-pretraining architecture ablation.
- Compact DINO tests narrow **internal** attention/MLP branches with fixed residual width, depth and output head; it does not establish the effect of neck depth or output resolution.
- DiT/DC-AE tests retaining the original output-decoder family with a full-width transformer interface; it does not remove the conditioning latent bottleneck or isolate one scalar architectural change under identical initialization history.

Do not claim proven causes for noisy learning, angular jaggedness, bottleneck damage, superiority of one pretraining method, or statistical significance solely from these variants. Report observed results separately from the rationale.

## 10. Evaluation context and implementation sources

First-allocation SS50 evaluations use the same **50 scene/frame pairs in identical order and matching seeds**, checked across saved records. Final first-run checkpoints are the steps in Section 8. The historical “final DINO” comparison is **2773475 step 19,914**, whereas its best-F1 intermediate checkpoint is **step 15,000**; always identify which one is used. Shared-data comparison does not remove training-history differences.

Foreground geometry metrics can have different valid-box denominators due to empty generated/GT point sets. Legacy RadarGen target-cell mark MAEs differ from current detection-wise mark MAEs. Entry-node inference timings are not hardware-controlled latency measurements. Bold table entries indicate best means, not significance.

Primary local implementation sources:

- [DINO, LoRA, neck and standard head](../../radargen/models/dino_ppp.py).
- [ResNet backbone substitution](../../radargen/models/resnet_ppp.py).
- [Initialization policy](../../radargen/training/ppp_initialization.py).
- [DiT output decoder adaptation](../../radargen/training/ppp_decoder.py).
- [Conditioning-only DiT](../../diffusion/model/nets/radargen_ppp.py) and [joint-stream blocks](../../diffusion/model/nets/radargen.py).
- [DC-AE stage definitions](../../diffusion/model/dc_ae/efficientvit/models/efficientvit/dc_ae.py).
- [Continuous-detection losses](../../radargen/losses/continuous.py) and [training integration](../../scripts/train_ppp.py).
- Production configs: [random DINO](../../configs/ablation_dino_random_init_train_4nodes.yaml), [ResNet](../../configs/ablation_resnet50_backbone_train_4nodes.yaml), [compact DINO](../../configs/ablation_dino_neck_compact_train_4nodes.yaml), [DiT](../../configs/ablation_dit_dcae_no_bottleneck_train_4nodes.yaml).
- Per-model implementation/test/job records: [random DINO](ablation_dino_random_init.md), [ResNet](ablation_resnet50_backbone.md), [compact DINO](ablation_dino_neck_compact.md), [DiT](ablation_dit_dcae_no_bottleneck.md).
- Result comparisons: [four-model table](SS50_four_model_comparison.md), [ResNet versus RadarGen](SS50_resnet50_vs_radargen.md).

Pretrained asset provenance:

- DINO: `/e/project1/nxtaim-1/huber7/pretrained/vit_small_patch16_dinov3.lvd1689m/model.safetensors` (compact model; random model deliberately loads none).
- ResNet: `output/pretrained_models/resnet50/resnet50-11ad3fa6.pth`, official ImageNet1K V2 asset, SHA-256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.
- SANA: `/e/project1/nxtaim-1/huber7/pretrained/Sana_600M_512px/checkpoints/Sana_600M_512px_MultiLing.pth`.
- DC-AE: `/e/project1/nxtaim-1/huber7/pretrained/dc-ae-f32c32-sana-1.1/`.
- Gemma: `/e/project1/nxtaim-1/huber7/pretrained/gemma-2-2b-it`; empty-embedding asset `output/pretrained_models/null_embed_diffusers_gemma-2-2b-it_300token_2304.pth`.
