# Related Work & Novelty Scan — Dust-Q (task T0-lit)

**Fetched 2026-10-11.** Every reference below was verified on 2026-10-11 by fetching the paper's abstract page, publisher/proceedings page, or Crossref metadata record. Abstracts were read at minimum; references marked **[full text read]** had large portions of the full text read. Anything that could not be independently verified is tagged **[unverified]** or **[partially verified]** with an explanation. No numbers, identifiers, or venues below are invented; where a field could not be confirmed it is left out or marked.

Our project: training quantized transformers (int8 and ternary weights, low-bit activations) **from scratch** with Dust, a zeroth-order (ZO) method that perturbs **activations** (per-token node perturbation, "virtual population"), vs. backprop+STE and a Gaussian-smoothed STE control, at 7M/38M params and 1M/10M tokens.

Two axes used throughout: **(a) perturbation space** — weights vs. activations/outputs; **(b) task** — from-scratch pretraining vs. fine-tuning/adaptation.

---

## Quick-reference table

| # | Method | Perturbation space | Task | STE | Quantization |
|---|--------|--------------------|------|-----|--------------|
| 1 | Dust (Q Labs 2026) | activations (per-token) | from-scratch pretraining | avoided (no backward) | none |
| 2 | QuZO (EMNLP 2025) | weights (low-bit stochastic) | fine-tuning | avoided explicitly | yes (int4/int8/fp8 forward) |
| 3 | QZO (ICLR 2026) | weights (via continuous quant. scale) | fine-tuning | avoided | yes (int4 etc.) |
| 4 | ZeroQAT (arXiv 2025) | weights | quantized (continued) pre-training + fine-tuning of pretrained LLMs | avoided explicitly | yes (weights **and** activations, W2A16/W4A4/W6A6) |
| 5 | HQZO (ICIC/LNCS 2026) | weights (Hessian-aware) | fine-tuning | avoided (ZO) | yes (low bit-width forward) |
| 6 | FOGZO (NeurIPS 2025) | weights (first-order-guided noise) | quantization-aware **pre-training** | **uses STE** (+ZO correction) | yes |
| 7 | ZOQO (ICASSP 2025) | weights (sign-based) | fine-tuning + adversarial attacks | avoided | yes (quantized params **and** ops) |
| 8 | MeZO (NeurIPS 2023) | weights | fine-tuning | avoided | none |
| 9 | EGGROLL (arXiv 2025/26) | weights (low-rank) | pretraining + post-training + RL | avoided (ES) | integer datatypes in one experiment [abstract-level] |
| 10 | ES at Scale (ICML 2026) | weights | full-parameter fine-tuning | avoided (ES) | none |
| 11 | Allaire et al. (ICPRAM 2025) | weights | from-scratch pretraining (lightweight) | avoided | none |
| 12 | KronZO (SN Comput. Sci. 2026) | weights (Kronecker) | from-scratch pretraining | avoided | none |
| 13 | Werfel et al. (NIPS 2003) | activations **and** weights (analysis) | from-scratch (linear nets) | n/a (perturbation methods) | none |
| 14 | Widrow & Lehr (1990) | activations/output (Madaline II trial perturbation) | small nets, from scratch | n/a | none |
| 15 | Ren et al. (ICLR 2023) | activations (activity perturbation, forward-mode AD) | from-scratch training (vision) | avoided (forward gradients) | none |
| 16 | BitNet b1.58 (arXiv 2024) | n/a (backprop) | from-scratch pretraining | uses STE | ternary weights |
| 17 | BitNet a4.8 (arXiv 2024) | n/a (backprop) | from-scratch + short continue-train | uses STE | ternary weights + 4-bit activations |
| 18 | BitNet v2 (arXiv 2025) | n/a (backprop) | from-scratch (+continue-train to 4-bit) | uses STE | ternary weights + INT8/INT4 activations |
| 19 | DSQ (ICCV 2019) | n/a (backprop) | from-scratch training (vision) | replaced by soft-quantizer derivative | yes (low-bit weights+activations) |
| 20 | QIL (CVPR 2019) | n/a (backprop) | from-scratch training (vision) | STE + learned intervals | yes |
| 21 | LQ-Nets (ECCV 2018) | n/a (backprop) | from-scratch training (vision) | STE + learned quantizers | yes |
| 22 | LSQ (ICLR 2020) | n/a (backprop) | from-scratch training (vision) | STE + learned step size | yes |
| 23 | Li et al. (NIPS 2017) | n/a (backprop) | from-scratch quantized training | analyzes STE (BinaryConnect) & stochastic rounding | yes |
| 24 | GFT (arXiv 2024/25) | weights (discrete stochastic search) | from-scratch training (vision) | avoided (no gradients at all) | yes (low-bit weights) |
| 25 | QZO-FF (NeurIPS 2024) | weights (quantized perturbations, forward gradients) | on-device fine-tuning/adaptation | avoided | yes (fixed-point 16w/8a) |
| 26 | DeepZero (ICLR 2024) | weights (coordinatewise) | from-scratch training (vision) | avoided | none |
| 27 | SOMA (arXiv Sept 2026) | weights (SPSA) | from-scratch pretraining (LSTM experts) | avoided | none |
| 28 | CAQ-ZO (arXiv May 2026) | weights (queries through quantized forward) | fine-tuning | avoided | yes (NF4 forward) |
| 29 | Hiratani et al. (NeurIPS 2022) | activations | from-scratch (theory/analysis) | n/a | none |
| 30 | Zoop (ICML 2025 workshop) | outputs | fine-tuning | hybrid (local derivatives) [partially verified] | none |
| 31 | HiZOO (ICLR 2025) | weights (Hessian-informed) | fine-tuning | avoided | none |

---

## 1. The base method

### [1] Dust — activation-space ZO pretraining
**Citation.** Samip Dahal, Bishwas Mandal, Serdar Gülbahar, Akshay Vegesna. "Dust: Pretraining Transformers Without Backpropagation." Q Labs Research, October 2026. Web: https://qlabs.sh/research/dust (appendix at /research/dust/appendix.html). Code: https://github.com/qlabs-eng/dust (MIT; minimal implementation, `dust.py` + `model.py` + backprop SGD baseline). No arXiv posting found as of 2026-10-11.

**What it does** [full text read]. Dust adds Gaussian noise to the output of each linear layer, independently at every token; each token's noise is rewarded by its centered loss reduction (plus decayed future reductions for attention internals), and the reward-weighted noise averaged over draws forms an estimated output error whose outer product with the cached layer input is the weight-gradient estimate — the same outer product backprop forms, with the output error coming from a "virtual population" (one population member per token) instead of the chain rule. It trains GPT-style transformers **from scratch** on FineWeb (4096-token BPE, 16k-token batch, one epoch, SGD with momentum plus an Adam section; base 8 layers × 512 width; token budgets 100k–20M; populations 64–16k draws; models 2M–243M params). Findings: competitive with and in several settings below tuned backprop (e.g., fitted limit 4.431 vs backprop 4.633 test loss at 20M tokens, with the fit still falling); 10³–10⁴× more efficient than its EGGROLL-Transformer implementation by extrapolation; larger models are more population-efficient, not less; gradient cosine to backprop follows cos(K)=c_max/√(1+c/K) and stays well aligned up to 1B-token checkpoints. The paper explicitly does not claim compute efficiency ("We do not attempt to make it compute-efficient enough to replace backprop today").

**Position.** Perturbation: activations. Task: from-scratch pretraining. STE: not used (no backward pass). Quantization: none.

---

## 2. Weight-space ZO + quantized fine-tuning family (the QuZO/QZO/ZeroQAT/HQZO line)

### [2] QuZO
**Citation.** Jiajun Zhou, Yifan Yang, Kai Zhen, Ziyue Liu, Yequan Zhao, Ershad Banijamali, Athanasios Mouchtaris, Ngai Wong, Zheng Zhang. "QuZO: Quantized Zeroth-Order Fine-Tuning for Large Language Models." In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025), pages 5341–5359, Suzhou, China, November 4–9, 2025. Association for Computational Linguistics. DOI: 10.18653/v1/2025.emnlp-main.271. arXiv:2502.12346. Links: https://aclanthology.org/2025.emnlp-main.271/ , https://arxiv.org/abs/2502.12346 [full text read in part: abstract, algorithm, related work, LoRA section].

**What it does.** ZO fine-tuning of LLMs through low-precision (4- or 8-bit, also FP8) forward passes: it quantizes the pretrained model's weights (frozen, at low precision) and estimates gradients with low-bit stochastic perturbations of the quantized weights evaluated by two quantized forward passes. It explicitly **avoids the error-prone low-precision straight-through estimator** and instead uses optimized stochastic rounding to mitigate the increased bias; it reports results comparable to first-order methods in FP8 and superior accuracy in INT8 and INT4, with 2.94×–5.47× memory reduction vs. quantized first-order methods during LLaMA-7B fine-tuning (arXiv v1 reports 2.94× for LLaMA2-7B), and supports both full-model and LoRA settings (quantized low-rank matrices on a frozen quantized base).

**Position.** Perturbation: weights (quantized). Task: fine-tuning. STE: avoided explicitly (a core claim). Quantization: yes — low-bit forward passes, INT4/INT8/FP8.

### [3] QZO
**Citation.** Sifeng Shang, Jiayi Zhou, Chenyu Lin, Minxian Li, Kaiyang Zhou. "Fine-tuning Quantized Neural Networks with Zeroth-order Optimization." ICLR 2026 (accepted; per arXiv comments field). arXiv:2505.13430 (v1 May 2025, v3 Feb 2026). https://arxiv.org/abs/2505.13430 [abstract + partial text read].

**What it does.** Eliminates gradients and optimizer states via ZO and weight memory via quantization (bfloat16 → int4); observing that perturbing discrete quantized weights is infeasible (precision gap would force dequantize/re-quantize), QZO **perturbs the continuous quantization scale** for gradient estimation and uses directional-derivative clipping for stability. It is orthogonal to scalar- and codebook-based PTQ, reduces total memory by more than 18× vs. 16-bit full-parameter fine-tuning for 4-bit LLMs, and fine-tunes Llama-2-13B within a single 24GB GPU. Its related-work section identifies the shared paradigm of prior ZO-quantized work (quantizing perturbation noises), citing Feng et al. 2024, QuZO (Zhou et al. 2025), and ZOQO (Bar & Giryes 2025).

**Position.** Perturbation: weights (via the continuous scale). Task: fine-tuning. STE: avoided. Quantization: yes — int4 weights.

### [4] ZeroQAT
**Citation.** Qitao Tan, Xiaoying Song, Jin Lu, Guoming Li, Jun Liu, Lingzi Hong, Caiwen Ding, Jundong Li, Xiaoming Zhai, Shaoyi Huang, Wei Niu, Geng Yuan. "End-to-End On-Device Quantization-Aware Training for LLMs at Inference Cost." arXiv:2509.00031 (v1 21 Aug 2025, v2 29 Sep 2025). No venue listed on arXiv as of 2026-10-11. https://arxiv.org/abs/2509.00031 [full text read].

**What it does.** ZeroQAT is a ZO optimization-based QAT framework supporting **both weight and activation quantization**: it maintains full-precision (latent) weights, evaluates quantized forward passes, and estimates gradients with forward-only finite differences — so it "does not require the straight-through estimator (STE)," with a theoretical appendix analyzing STE bias and noting the ZO estimator is unbiased w.r.t. the gradient of a smoothed quantized objective. It jointly learns channel-wise activation/weight smoothing parameters, step size, zero-point, and clipping coefficients; a lightweight variant (for fine-tuning only — it degrades for quantized pre-training) freezes and pre-quantizes everything except attention Q/V. **Crucial nuance:** its "quantized pre-training" experiments apply ZO-QAT to *pretrained* Llama-1/2 (7B/13B) and OPT models with WikiText2+C4 segments under W2A16(g128)/W4A4/W6A6 — i.e., continued language-model training under quantization, **not training from scratch**. Results: outperforms PTQ/QAT baselines (e.g., 5.1% average gain over five zero-shot tasks at 2-bit weight-only; W4A4 SST-2 ≈ 88% on OPT fine-tuning), fine-tunes a 13B model at 2–4 bits on a single 8GB GPU, and a 6.7B model on a OnePlus 12 smartphone. It positions QuZO and QZO as weight-only-scenario methods and claims to be the first end-to-end QAT framework supporting both low-bit weight and activation on-device quantization.

**Position.** Perturbation: weights. Task: quantization-aware continued pre-training and fine-tuning of pretrained LLMs (not from scratch). STE: avoided explicitly. Quantization: yes — weights and activations, 2–6 bits.

### [5] HQZO
**Citation.** Zhangchi Wang, Yidu Wu, Yijin Huang, Pujin Cheng, Qinghai Guo, Xiaoying Tang. "Hessian-Aware Zeroth-Order Optimization for Quantized Large Language Models." In: *Advanced Intelligent Computing Technology and Applications* (Lecture Notes in Computer Science), pages 391–403. Springer Nature Singapore, published online 18 July 2026. 22nd International Conference on Intelligent Computing (ICIC 2026), Toronto, Canada, July 22–26, 2026. DOI: 10.1007/978-981-92-3394-6_33. ISBN 978-981-92-3394-6 (electronic). Link: https://doi.org/10.1007/978-981-92-3394-6_33 . Verified via Crossref metadata record; Springer page did not render for our fetcher. Abstract read in part via search snippets.

**What it does.** Proposes HQZO, a Hessian-aware zeroth-order optimization framework "tailored for fine-tuning quantized LLMs via low bit-width forward propagation" (per the abstract snippet), building on the ZO line (MeZO, ZO-signSGD, QuZO, QZO) and the Hessian-aware ZO black-box-attack literature (Ye et al., arXiv:1812.11377).

**Position.** Perturbation: weights (Hessian-aware ZO). Task: fine-tuning of quantized LLMs. STE: avoided (ZO). Quantization: yes — low bit-width forward propagation. *(Existence, authors, venue, pages, and DOI fully verified; method summary limited by partial abstract access.)*

### [6] ZOQO
**Citation.** Noga Bar, Raja Giryes. "ZOQO: Zero-Order Quantized Optimization." ICASSP 2025 (accepted; per arXiv comments field). arXiv:2501.06736. https://arxiv.org/abs/2501.06736 [abstract + partial text read].

**What it does.** A zero-order optimization method designed for models with quantized parameters **and** quantized operations: it uses zero-order approximations of the gradient sign and adapts the learning process to maintain parameter quantization without full-precision gradient calculations, making all updates carried out in quantized form. Evaluated on LLM fine-tuning (including LoRA) and black-box adversarial attacks, achieving competitive performance vs. full-precision methods.

**Position.** Perturbation: weights (sign-based). Task: fine-tuning (+ adversarial-attack optimization). STE: avoided (no FP gradients at all). Quantization: yes — quantized parameters and operations.

### [7] CAQ-ZO (newer member of the family)
**Citation.** Yao Shu, Zilin Zhu. "Compander-Aligned Query Geometry for Quantized Zeroth-Order Optimization." arXiv:2605.10673 (submitted 11 May 2026). https://arxiv.org/abs/2605.10673 [abstract read].

**What it does.** Studies ZO optimization when the forward evaluation itself is quantized (nonuniform companding quantizers, e.g., NF4): a quantized ZO query is not a continuous finite difference followed by storage rounding — the query chooses endpoints, the engine rounds them, and the loss difference is measured along the rounded chord. CAQ-ZO forms one-grid-step Rademacher stencils in compander space, with stationarity bounds showing generic off-grid queries retain a Δ²/μ² residual that CAQ-ZO removes; improves NF4 Qwen/Llama fine-tuning under the same quantizer and budget.

**Position.** Perturbation: weights (queries through a quantized forward). Task: fine-tuning. STE: avoided. Quantization: yes — low-bit (NF4) forward evaluation.

### [8] QZO-FF — quantized forward gradients on device
**Citation.** Chen Feng, Shaojie Zhuo, Xiaopeng Zhang, Ramchalam Kinattinkara Ramakrishnan, Zhaocong Yuan, Andrew Zou Li. "Stepping Forward on the Last Mile." Advances in Neural Information Processing Systems 37 (NeurIPS 2024). arXiv:2411.04036. https://arxiv.org/abs/2411.04036 [abstract + method sections read].

**What it does.** Investigates on-device training with fixed-point forward gradients: QZO-FF quantizes the weight perturbation, the forward-gradient estimate (sign-m-SPSA from two forward calls), and the weight update into fixed point (16-bit weights; 8-bit perturbations/gradients), so training runs on fixed-point inference engines at near-inference memory. Finds 16-bit weights necessary (8-bit weights fail to converge); benchmarks few-shot and cross-domain adaptation across vision and audio (ResNet, ViT-tiny, CRNN, AST) at 16w8a.

**Position.** Perturbation: weights (quantized). Task: on-device fine-tuning/adaptation. STE: avoided (forward-gradient, no backprop). Quantization: yes — fixed-point weights/perturbations/gradients.

---

## 3. Weight-space ZO fine-tuning and ES at scale (unquantized)

### [9] MeZO
**Citation.** Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, Sanjeev Arora. "Fine-Tuning Language Models with Just Forward Passes." Advances in Neural Information Processing Systems 36 (NeurIPS 2023, oral). arXiv:2305.17333. https://arxiv.org/abs/2305.17333 [abstract read]. *Note: the correct arXiv ID is 2305.17333 — use this in all drafts.*

**What it does.** Adapts classical ZO-SGD to operate in-place with two forward passes, fine-tuning LMs with the same memory footprint as inference (30B model on one A100 80GB); achieves performance comparable to backprop fine-tuning across classification, multiple-choice, and generation tasks up to 66B with up to 12× memory reduction, compatible with LoRA/prefix tuning, and able to optimize non-differentiable objectives.

**Position.** Perturbation: weights. Task: fine-tuning. STE: avoided (no backprop). Quantization: none.

### [10] Evolution Strategies at Scale
**Citation.** Xin Qiu, Yulu Gan, Conor F. Hayes, Qiyao Liang, Yinggan Xu, Roberto Dailey, Elliot Meyerson, Babak Hodjat, Risto Miikkulainen. "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning." ICML 2026 main conference (per arXiv comments field). arXiv:2509.24372 (v1 Sep 2025, v3 Jul 2026). https://arxiv.org/abs/2509.24372 [abstract read].

**What it does.** Demonstrates the first successful application of ES to full-parameter fine-tuning of billion-parameter LLMs without dimensionality reduction, showing ES outperforms established RL implementations on tolerance to long-horizon and delayed rewards, robustness across base LLMs, reduced reward hacking, and training stability — a backprop-free post-training paradigm.

**Position.** Perturbation: weights. Task: full-parameter fine-tuning (post-training). STE: avoided (backprop-free ES). Quantization: none.

### [11] HiZOO (sibling of HQZO, unquantized)
**Citation.** Yanjun Zhao, Sizhe Dang, Haishan Ye, Guang Dai, Yi Qian, Ivor Tsang. "Second-Order Fine-Tuning without Pain for LLMs: A Hessian Informed Zeroth-Order Optimizer." ICLR 2025. arXiv:2402.15173. https://proceedings.iclr.cc/paper_files/paper/2025/hash/6bf82cc56a5fa0287c438baa8be65a70-Abstract-Conference.html [abstract read via proceedings page].

**What it does.** HiZOO, a diagonal-Hessian-informed ZO optimizer for LLM fine-tuning (claimed first to leverage the diagonal Hessian in ZOO for this setting), reduces training steps and improves accuracy over MeZO (e.g., 8× speedup on SST-2) across models from 350M to 66B; HiZOO-L cuts Hessian memory to ~10% of MeZO's.

**Position.** Perturbation: weights. Task: fine-tuning. STE: avoided. Quantization: none (quantization named only as future work).

### [12] Zoop — output perturbation [partially verified]
**Citation.** Xixi Hu, Bo Liu, Qiang Liu, Xiaocong Du, Bhargav Bhushanam, Louis Feng, Chengyue Gong, Kaizhao Liang. "Zoop it! Efficient Zero-Order Optimization with Output Perturbation." ICML 2025 Workshop on Tiny Titans: The next wave of On-Device Learning for Foundation Models. OpenReview: https://openreview.net/forum?id=Tc8vFyRhPO . **Bibliographic record verified via Dust's reference list**; the OpenReview page returned a bot-verification wall on our 2026-10-11 fetch attempts, so the abstract was not independently read. Per Dust's related-work section: "uses output perturbations for language model fine tuning, converting estimated output gradients into parameter updates with local derivatives."

**Position.** Perturbation: outputs. Task: fine-tuning. STE: hybrid — uses local derivatives to convert output-gradient estimates into parameter updates [per Dust]. Quantization: none.

---

## 4. Weight-space ZO pretraining (from scratch) — exists, but weak or new

### [13] Allaire et al., ICPRAM 2025
**Citation.** Nathan Allaire, Mahsa Ghazvini Nejad, Sébastien Le Digabel, Vahid Partovi Nia. "Zeroth Order Optimization for Pretraining Language Models." In Proceedings of the 14th International Conference on Pattern Recognition Applications and Methods (ICPRAM 2025) — Volume 1, pages 113–121. SciTePress, Porto, Portugal, February 23–25, 2025. ISBN 978-989-75-7306. ISSN 2184-4313. DOI: 10.5220/0013261100003905. Open-access PDF: https://www.scitepress.org/Papers/2025/132611/132611.pdf . Received the ICPRAM 2025 Best Industrial Paper Award (per GERAD announcement, gerad.ca/en/posts/1228). [abstract read; PDF fetched]

**What it does.** Applies weight-perturbation ZO optimization to pretraining lightweight language models from scratch; concludes that ZO "cannot be readily applied" to pretraining and that **the curse of dimensionality is the main obstacle**, paving the way toward modified ZO methods for pretraining; builds theoretical connections between second-, first-, and zeroth-order methods. Dust's related work describes it as examining "the difficulty of training transformers from scratch with weight perturbations."

**Position.** Perturbation: weights. Task: from-scratch pretraining (lightweight models). STE: avoided. Quantization: none.

### [14] KronZO
**Citation.** Nathan Allaire, Sébastien Le Digabel, Dominique Orban, Vahid Partovi Nia. "Zeroth-Order Kronecker Optimization for Pretraining Language Models." SN Computer Science, 7(2), Article 162, published online 2 February 2026. DOI: 10.1007/s42979-025-04704-9. GERAD technical report G-2025-44 (July 2025, revised September 2025): https://www.gerad.ca/en/papers/G-2025-44 . Verified via Crossref and GERAD. [abstract read via GERAD]

**What it does.** Observes that gradient singular-value spectra during pretraining are spread across many directions, so low-rank ZO perturbations discard informative components; KronZO explores a **full-rank** search subspace via Kronecker-structured perturbations with state-of-the-art storage compression plus criterion-driven directional selective updates. Pretraining GPT-2 Small **from scratch** on OpenWebText achieves markedly lower training loss than all previous ZO baselines at less GPU memory — though still trailing first-order methods in final loss.

**Position.** Perturbation: weights (Kronecker, full-rank). Task: from-scratch pretraining. STE: avoided. Quantization: none.

### [15] DeepZero — ZO deep learning from scratch (vision)
**Citation.** Aochuan Chen, Yimeng Zhang, Jinghan Jia, James Diffenderfer, Jiancheng Liu, Konstantinos Parasyris, Yihua Zhang, Zheng Zhang, Bhavya Kailkhura, Sijia Liu. "DeepZero: Scaling up Zeroth-Order Optimization for Deep Model Training." ICLR 2024. arXiv:2310.02025. https://arxiv.org/abs/2310.02025 [abstract read].

**What it does.** Scales ZO to DNN training **from scratch** via coordinatewise gradient estimation (beating vector-wise), a sparsity-induced ZO training protocol, feature reuse, and forward parallelization; achieves SOTA accuracy on ResNet-20/CIFAR-10, approaching first-order training performance for the first time.

**Position.** Perturbation: weights (coordinatewise). Task: from-scratch training (vision). STE: avoided. Quantization: none.

### [16] SOMA — concurrent ZO pretraining work
**Citation.** Francois Chaubard, Mykel J. Kochenderfer, Chris Ré. "Scaling Zero-Order Pretraining through Model Sharding." arXiv:2609.37899 (submitted 29 September 2026). https://arxiv.org/abs/2609.37899 [abstract read].

**What it does.** SOMA (Sharded Optimization Mixture of Assemblies) trains LSTM experts independently on N data clusters using SPSA without exchanging gradients, activations, or optimizer state; the separable loss removes cross-expert perturbation noise (proved to reduce relative gradient variance to ≈1/N). At 8.44M parameters, SOMA N=2 with 64 perturbations reaches 1.76 test nats/byte vs. 2.00–2.11 for monolithic SPSA and 2.21 for EGGROLL; large ensembles give inference-throughput benefits (9.19× at N=256 with top-4 routing) at 59.9× the aggregate training compute.

**Position.** Perturbation: weights (SPSA). Task: from-scratch pretraining (sharded LSTM byte-LMs). STE: avoided. Quantization: none. *Does not cite Dust in its abstract; contemporaneous with Dust (Sept 29 vs. Oct 2026).*

### [17] EGGROLL — weight-space ES, our weight-space foil
**Citation.** Bidipta Sarkar, Mattie Fellows, Juan Agustin Duque, Alistair Letcher, Antonio León Villares, Anya Sims, Clarisse Wibault, Dmitry Samsonov, Dylan Cope, Jarek Liesen, Kang Li, Lukas Seier, Theo Wolf, Uljad Berdica, Valentin Mohl, Alexander David Goldie, Aaron Courville, Karin Sevegnani, Shimon Whiteson, Jakob Nicolaus Foerster. "Evolution Strategies at the Hyperscale." arXiv:2511.16652 (v1 20 Nov 2025, v2 16 Feb 2026). Website: https://eshyperscale.github.io/ . https://arxiv.org/abs/2511.16652 [abstract read].

**What it does.** Structures weight perturbations as rank-r matrices to raise arithmetic intensity (~100× training speedup for billion-parameter models at large populations, up to 91% of batch-inference throughput), with theory on Gaussian-ES convergence in high dimensions. Experiments: (1) "enables the stable pretraining of nonlinear recurrent language models that operate purely in integer datatypes," (2) competitive with GRPO for post-training LLMs on reasoning, (3) matches vanilla ES in tabula-rasa RL. **[unverified beyond abstract]** — the nature of the integer-datatype experiment (whether weights/activations are quantized from scratch, or continued training) is not established from the abstract alone; full-text check required (see Overlap risks).

**Position.** Perturbation: weights (low-rank). Task: pretraining + post-training + RL. STE: avoided (ES). Quantization: integer datatypes in one pretraining experiment [abstract-level only].

---

## 5. Activation/output-space perturbation line (our axis)

### [18] Werfel, Xie, Seung — node perturbation theory
**Citation.** Justin Werfel, Xiaohui Xie, H. Sebastian Seung. "Learning Curves for Stochastic Gradient Descent in Linear Feedforward Networks." Advances in Neural Information Processing Systems 16 (NIPS 2003). https://proceedings.neurips.cc/paper_files/paper/2003/hash/f8b932c70d0b2e6bf071729a4fa68dfc-Abstract.html [abstract + core sections read].

**What it does.** Derives quantitative learning curves for three online training methods on a linear perceptron: direct gradient descent, **node perturbation**, and weight perturbation. Online gradient descent is faster by a factor of M (output dimension) than node perturbation, which is in turn faster by a factor of N (input dimension) than weight perturbation; the maximum learning rate of the stochastic methods scales inversely with the dimensionality of the injected noise.

**Position.** Perturbation: activations (node) and weights (compared). Task: from-scratch training (linear nets). STE: n/a. Quantization: none.

### [19] Widrow & Lehr — historical root
**Citation.** Bernard Widrow, Michael A. Lehr. "30 Years of Adaptive Neural Networks: Perceptron, Madaline, and Backpropagation." Proceedings of the IEEE, 78(9):1415–1442, September 1990. DOI: 10.1109/5.58323. PDF: https://www-isl.stanford.edu/~widrow/papers/j199030years.pdf [abstract + text read].

**What it does.** Reviews supervised training algorithms for feedforward networks (Perceptron rule, LMS, Madaline rules I–III, backpropagation) under the "minimal disturbance" principle; Madaline Rule II adapts multilayer networks by *trial perturbation* of element outputs (a precursor of node perturbation), and Madaline Rule III is shown to be mathematically equivalent to backpropagation. Dust cites this as the node-perturbation origin.

**Position.** Perturbation: activations/outputs (MRII trial flips). Task: from-scratch training of small nets. STE: n/a. Quantization: none.

### [20] Ren et al. — activity perturbation + forward gradients
**Citation.** Mengye Ren, Simon Kornblith, Renjie Liao, Geoffrey Hinton. "Scaling Forward Gradient With Local Losses." ICLR 2023. arXiv:2210.03310. OpenReview: https://openreview.net/forum?id=JxpBP1JM15- [abstract + sections read].

**What it does.** Forward-gradient learning estimates a noisy directional derivative using only forward passes; this paper shows that **applying perturbations to activations rather than weights substantially reduces estimator variance** (the dimensionality argument), and scales the approach with many local greedy losses (block-, patch-, channel-group-wise) plus a LocalMixer architecture; it matches backprop on MNIST/CIFAR-10 and significantly outperforms prior backprop-free algorithms on ImageNet (without fully matching backprop at larger scale).

**Position.** Perturbation: activations (activity perturbation, forward-mode AD). Task: from-scratch training (vision). STE: avoided (no backward passes for gradient estimation). Quantization: none.

### [21] Hiratani et al. — node perturbation stability/scalability
**Citation.** Naoki Hiratani, Yash Mehta, Timothy Lillicrap, Peter E. Latham. "On the Stability and Scalability of Node Perturbation Learning." Advances in Neural Information Processing Systems 35 (NeurIPS 2022). https://proceedings.neurips.cc/paper_files/paper/2022/hash/cf38eb1549024cce4b3d2c1bb87a6c27-Abstract-Conference.html [abstract read].

**What it does.** Analyzes node-perturbation learning dynamics analytically (deep linear networks) and numerically (deep nonlinear networks): learning time depends only weakly on hidden-layer size, but node perturbation is **unstable under model mismatch** (weight diffusion → very large weights), suppressible by weight normalization at the cost of bias in the learning rule.

**Position.** Perturbation: activations. Task: from-scratch training (analysis). STE: n/a. Quantization: none.

**Brief mention (same line).** "Node Perturbation Can Effectively Train Multi-Layer Neural Networks" (arXiv:2310.00965) — title/abstract content surfaced repeatedly in searches (activity-based node perturbation, directional-derivative reframing); **[authors not captured in this pass]**. GEMINI: Yann Le Cun, Conrad C. Galland, Geoffrey E. Hinton, "GEMINI: Gradient estimation through matrix inversion after noise injection," NIPS 1988, pp. 141–148 — injects noise into the first hidden layer and recovers layerwise gradient estimates via matrix inversion (bibliographic record verified via Dust's reference list).

---

## 6. From-scratch quantized pretraining with backprop/STE (BitNet line) and STE hybrids

### [22] BitNet b1.58
**Citation.** Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei. "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." arXiv:2402.17764 (27 Feb 2024; arXiv comment: "Work in progress"). https://arxiv.org/abs/2402.17764 [abstract read]. *No published venue verified — cite as arXiv preprint.*

**What it does.** Introduces BitNet b1.58, where **every weight is ternary {-1, 0, 1}**; it matches full-precision (FP16/BF16) transformer LLMs of the same size and training-token count in perplexity and end-task performance while being significantly more cost-effective in latency, memory, throughput, and energy, and argues for a new scaling law/recipe and 1-bit-oriented hardware.

**Position.** Perturbation: n/a (backprop). Task: from-scratch pretraining. STE: used (BitLinear QAT). Quantization: ternary weights (high-precision activations).

### [23] BitNet a4.8 — 4-bit activations exist (with backprop)
**Citation.** Hongyu Wang, Shuming Ma, Furu Wei. "BitNet a4.8: 4-bit Activations for 1-bit LLMs." arXiv:2411.04965 (7 Nov 2024). https://arxiv.org/abs/2411.04965 [abstract + sections read]. *No published venue verified — arXiv preprint ("Work in progress" per Microsoft Research page).*

**What it does.** Enables 4-bit activations for 1-bit (ternary-weight) LLMs via hybrid quantization and sparsification: 4-bit (INT4/FP4) inputs to attention/FFN, with top-K sparsified 8-bit intermediate states; trained with a two-stage recipe — first 8-bit activations + ReLU² GLU, then the hybrid scheme — adapting a BitNet b1.58 model to low-bit activations with few training tokens at the end of training. Performance comparable to b1.58 at equivalent training cost; ~55% activated parameters; 3-bit KV cache.

**Position.** Perturbation: n/a (backprop). Task: from-scratch pretraining plus a short low-bit continue-train stage. STE: used. Quantization: ternary weights + 4-bit activations.

### [24] BitNet v2
**Citation.** Hongyu Wang, Shuming Ma, Furu Wei. "BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs." arXiv:2504.18415 (v1 Apr 2025, v2 Jun 2025; "Work in progress"). https://arxiv.org/abs/2504.18415 [abstract read].

**What it does.** Introduces H-BitLinear, applying an online Hadamard transformation before activation quantization to smooth outlier-heavy activation distributions into more Gaussian-like forms suitable for 4-bit; BitNet v2 **trained from scratch** with 1.58-bit weights and 8-bit activations matches BitNet b1.58, and it degrades minimally when trained natively with 4-bit activations (with a short continue-train from the 8-bit variant, per its comparison setup on RedPajama tokens).

**Position.** Perturbation: n/a (backprop). Task: from-scratch pretraining (+continue-train to 4-bit activations). STE: used. Quantization: ternary weights + INT8/INT4 activations.

### [25] FOGZO — the closest work on (pretraining × quantization × ZO-adjacent) ⚠ top overlap
**Citation.** Ningfeng Yang, Tor M. Aamodt. "Improving the Straight-Through Estimator with Zeroth-Order Information." Advances in Neural Information Processing Systems 38 (NeurIPS 2025), Main Conference Track. arXiv:2510.23926. Code: https://github.com/1733116199/fogzo . https://arxiv.org/abs/2510.23926 , https://papers.neurips.cc/paper_files/paper/2025/hash/f49d76cf84df83a611883c621c96d2d9-Abstract-Conference.html [abstract read; NeurIPS poster page read; public code inspected].

**What it does.** Observes that the STE provides high-quality *biased* gradients while ZO gradients are *unbiased* but expensive, and proposes FOGZO (First-Order-Guided Zeroth-Order Gradient Descent), which reduces STE bias at lower cost than ZO by mixing the normalized first-order gradient with random noise in the perturbation direction (per the public implementation) and spending two extra forward passes per iteration. In **Quantization-Aware Pre-Training** — versus STE at the same iteration count — FOGZO improves accuracy by 1–8% on DeiT Tiny/Small, 1–2% on ResNet 18/50, and 1–22 perplexity points on LLaMA models up to 0.3B parameters, and yields a 796× compute reduction vs. n-SPSA at equal loss on a 2-layer MLP/MNIST. NeurIPS materials describe FOGZO as "specialized for pre-training" and "the first partially-zeroth-order method to consistently outperform a first-order method STE for DNNs," improving both QuEST and LSQ when activations and weights are quantized.

**Position.** Perturbation: weights (first-order-guided noise). Task: **from-scratch quantization-aware pre-training**. STE: **used** (STE + ZO hybrid — still requires backprop for the first-order part; *not* a pure ZO method). Quantization: yes — quantized weights and activations.

### [26] GFT — gradient-free from-scratch training of quantized nets ⚠ overlap
**Citation.** Noa Cohen, Omkar Joglekar, Dotan Di Castro, Vladimir Tchuiev, Shir Kozlovsky, Michal Moshkovitz. "Gradient-Free Training of Quantized Neural Networks." arXiv:2410.09734 (v1 Oct 2024, v2 Sep 2025). https://arxiv.org/abs/2410.09734 [abstract + partial text read]. *No published venue verified — arXiv preprint.*

**What it does.** Proposes eliminating gradients altogether for training quantized networks: it proves that finding optimal weights in a quantized space is NP-hard even when the continuous problem is easy, then introduces GFT, a stochastic-search heuristic that computes a contribution matrix and moves only the most important weights to the adjacent quantized value aligned with the desired change. Achieves performance comparable to full-precision gradient-based training on standard datasets and architectures with up to 3× less energy and 5× fewer parameter updates. The paper distinguishes itself from ZO methods (and from ZOQO) by prioritizing *which* weights to update rather than updating all parameters.

**Position.** Perturbation: weights (discrete, subset updates; not population-based loss-difference estimation). Task: from-scratch training (standard vision datasets/architectures per the abstract). STE: avoided (no gradients at all). Quantization: yes — low-bit weights.

---

## 7. Smoothed/learned quantizer gradients — prior art for our "Gaussian-smoothed STE" baseline

### [27] DSQ
**Citation.** Ruihao Gong, Xianglong Liu, Shenghu Jiang, Tianxiang Li, Peng Hu, Jiazhen Lin, Fengwei Yu, Junjie Yan. "Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks." Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2019), pages 4851–4860 (CVF open access lists 4852–4861). DOI: 10.1109/ICCV.2019.00495. arXiv:1908.05033. https://arxiv.org/abs/1908.05033 [abstract read]. *Note: an early plan draft guessed arXiv ~1908.05046 — that ID is an unrelated condensed-matter physics paper; the correct ID is 1908.05033.*

**What it does.** Replaces the hard quantizer with a **differentiable soft quantizer** (piecewise tanh) that automatically evolves during training to gradually approximate standard quantization, providing more accurate gradients in backpropagation and jointly optimizing the clipping range; training low-bit networks with DSQ outperforms prior quantization methods, with a 2–4-bit ARM implementation achieving up to 1.7× speedup over an 8-bit inference framework.

**Position.** Perturbation: n/a (backprop). Task: from-scratch training (vision CNNs). STE: replaced by the soft-quantizer derivative (an alternative to the identity-STE). Quantization: yes — binary/uniform low-bit weights and activations.

### [28] QIL
**Citation.** Sangil Jung, Changyong Son, Seohyung Lee, Jinwoo Son, Jae-Joon Han, Youngjun Kwak, Sung Ju Hwang, Changkyu Choi. "Learning to Quantize Deep Networks by Optimizing Quantization Intervals with Task Loss." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2019), pages 4350–4359. arXiv:1808.05779. https://arxiv.org/abs/1808.05779 [abstract read via CVF open access].

**What it does.** Learns to quantize activations and weights via a trainable quantizer that parameterizes the quantization intervals and optimizes them by directly minimizing the task loss (quantization-interval learning); 4-bit quantized networks maintain full-precision accuracy with graceful degradation at 2–3 bits.

**Position.** Perturbation: n/a (backprop). Task: from-scratch training (vision). STE: STE-based training with learned quantizer parameters. Quantization: yes.

### [29] LQ-Nets
**Citation.** Dongqing Zhang, Jiaolong Yang, Dongqiangzi Ye, Gang Hua. "LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks." Proceedings of the European Conference on Computer Vision (ECCV 2018), pages 365–382. https://openaccess.thecvf.com/content_ECCV_2018/html/Dongqing_Zhang_Optimized_Quantization_for_ECCV_2018_paper.html [abstract read].

**What it does.** Jointly trains a quantized, bit-operation-compatible DNN and its associated quantizers (for both weights and activations, arbitrary-bit precision), rather than using fixed handcrafted uniform or logarithmic schemes; consistently improves accuracy across AlexNet/VGG/GoogLeNet/ResNet/DenseNet on CIFAR-10 and ImageNet.

**Position.** Perturbation: n/a (backprop). Task: from-scratch training (vision). STE: STE-based learned quantizers. Quantization: yes.

### [30] LSQ
**Citation.** Steven K. Esser, Jeffrey L. McKinstry, Deepika Bablani, Rathinakumar Appuswamy, Dharmendra S. Modha. "Learned Step Size Quantization." ICLR 2020. arXiv:1902.08153. https://arxiv.org/abs/1902.08153 [abstract read].

**What it does.** Learns each layer's quantizer step size through the task loss (with a gradient scaled to be sensitive to quantized state transitions), achieving (at publication) the highest ImageNet accuracy for 2/3/4-bit weights-and-activations models and training 3-bit models to full-precision baseline accuracy.

**Position.** Perturbation: n/a (backprop). Task: from-scratch training (vision). STE: STE-based with learned step size. Quantization: yes.

### [31] Training Quantized Nets — STE/SR theory
**Citation.** Hao Li, Soham De, Zheng Xu, Christoph Studer, Hanan Samet, Tom Goldstein. "Training Quantized Nets: A Deeper Understanding." Advances in Neural Information Processing Systems 30 (NIPS 2017). arXiv:1706.02379. https://arxiv.org/abs/1706.02379 [abstract read].

**What it does.** Theoretical analysis of two from-scratch quantized-training methods — BinaryConnect (STE with a latent full-precision copy) and stochastic rounding — deriving accuracy guarantees depending on discretization coarseness and showing that methods exploiting high-precision representations have a greedy-search phase that purely quantized training lacks.

**Position.** Perturbation: n/a (backprop). Task: from-scratch quantized training. STE: analyzed (BinaryConnect/STE + stochastic rounding). Quantization: yes.

**Brief mentions (same family).** EWGS: Junghyup Lee, Dohyung Kim, Bumsub Ham, "Network Quantization with Element-wise Gradient Scaling," arXiv:2104.00903 (abstract verified on arXiv; venue not independently verified) — adaptively rescales each STE gradient element by gradient sign and discretization error, an STE alternative rather than a smoother. Yin et al., "Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets," ICLR 2019, arXiv:1903.05662 (verified as a reference in HQZO's Crossref record) — theory of STE as a descent direction for activation-quantized two-layer networks. Malinovskii et al. 2024 on STE errors in low-bit formats is cited by QuZO **[not independently fetched]**. Relevance to us: the "Gaussian-smoothed STE" control has a long lineage of smoothed/learned-quantizer prior art (DSQ's evolving soft quantizer being the closest in spirit), and ZeroQAT's Appendix G already frames ZO as unbiased for a *smoothed* quantized objective — both must be cited when we motivate that control.

---

## 8. Novelty-niche searches (run 2026-10-11)

1. **"zeroth-order training quantized network from scratch"** → surfaced GFT (gradient-free quantized training from scratch, vision), DeepZero (ZO from scratch, unquantized vision), ZOQO/QuZO (fine-tuning), FOGZO (STE+ZO QAT pre-training), and the NIPS 2017 STE/SR theory. **No work found that trains quantized LMs from scratch with a ZO/ES method.**
2. **"evolution strategies binary ternary neural network training"** → mainstream binary/ternary training is backprop+STE throughout (Ternary Weight Networks, arXiv:1605.04711; Trained Ternary Quantization, arXiv:1612.01064; BNN/BinaryConnect cited therein); historical GA+NN work (Hinton & Nowlan, NIPS 1990) is unrelated to quantization. Nearest hits: EGGROLL's integer-datatype recurrent LM pretraining [abstract-level] and GFT's discrete search. **No ES-based ternary/binary LLM training found.**
3. **"forward gradient quantization"** → direct hit: QZO-FF (NeurIPS 2024, arXiv:2411.04036) — quantized fixed-point forward-gradient training for on-device adaptation. Also surfaced an IEEE listing "FF-INT8" (Forward-Forward INT8 training) **[listing seen, paper not fetched — unverified]**.
4. **"node perturbation quantized"** → nothing combining node perturbation with quantization. Node-perturbation literature remains unquantized (Werfel 2003; Hiratani et al. NeurIPS 2022; arXiv:2310.00965; Ren et al. 2023).
5. **"activation perturbation quantized training"** → no direct combination found. Nearest: "activation-informed/guided" ZO fine-tuning — ZO-Act (arXiv:2607.01125; title/abstract content seen via its arXiv HTML, **[authors not captured in this pass]**) and AGZO ("activation-guided zero-order optimization for LLM fine-tuning," Lin et al. 2026) **[unverified — seen only as a citation inside ZO-Act's related work]**. These use activations to *construct weight-space subspaces*; they do **not** perturb activations.
6. **"Dust quantization" / follow-ups citing Dust** → **no citing papers found as of 2026-10-11**; only press/discussion coverage (Hacker News thread; AI/TLDR release note dated 2026-10-05; digg; picx.dev). The paper is roughly one week old. SOMA (arXiv:2609.37899, submitted 2026-09-29) is contemporaneous ZO pretraining work and does not appear to build on Dust (its comparisons are vs. SPSA/EGGROLL).

---

## 9. Positioning

Our project sits at the intersection of two threads that have so far met only in weight space and only in fine-tuning. **Weight-space ZO + quantized fine-tuning is an established family** — QuZO (EMNLP 2025), QZO (ICLR 2026), ZeroQAT, HQZO (ICIC 2026), ZOQO (ICASSP 2025), and CAQ-ZO (2026) all adapt pretrained LLMs with forward-only updates through low-bit forwards, and several explicitly frame STE-avoidance as a core benefit (QuZO, ZeroQAT) — but every one of them perturbs weights (or quantization scales) of a model that already exists. **Weight-space ZO pretraining from scratch exists but is weak or brand new**: Allaire et al. (ICPRAM 2025) identified the curse of dimensionality as the obstacle, KronZO (SN Comput. Sci. 2026) narrows but does not close the gap to first-order methods, SOMA (Sept 2026) sidesteps the problem by sharding, and Dust's own measurements show weight-space ES (EGGROLL) needing 10³–10⁴× the population to match. **Activation-space ZO pretraining only just became competitive with Dust (Oct 2026)** — and Dust involves no quantization. Meanwhile, **from-scratch training of low-bit transformers exists only with backprop+STE**: the BitNet line (b1.58 ternary; a4.8 and v2 with 4-bit activations) trains quantized models from scratch with straight-through gradients, FOGZO (NeurIPS 2025) improves STE-based quantization-aware pre-training by injecting zeroth-order information while still using first-order STE gradients via backprop, and GFT (2024) trains small quantized vision networks by discrete stochastic search. The cell our work occupies — **activation-space perturbation (Dust's per-token virtual population) evaluated through a low-bit quantized forward pass, for pretraining transformers from scratch** — is, as far as our 2026-10-11 searches show, unclaimed; filling it also gives the first matched-budget, same-data comparison of a pure backprop-free ZO estimator against STE and a Gaussian-smoothed STE control for quantized from-scratch pretraining of language models.

---

## 10. Claim discipline

**We MAY claim (with "to our knowledge" hedges and after the overlap checks below):**
- To our knowledge, the first zeroth-order method that trains quantized transformers (low-bit weights **and** activations) **from scratch**, and the first to combine activation-space ZO credit assignment with a quantized forward pass in any setting.
- The first systematic comparison of a pure (backprop-free) activation-perturbation ZO estimator vs. STE vs. a Gaussian-smoothed STE control, at matched budgets, for from-scratch quantized LM pretraining.
- Our empirical findings at the scales we actually run (7M/38M params; 1M/10M tokens; int8/ternary weights, low-bit activations), including any analysis of how a quantized forward changes the noise-reward interaction in the Dust estimator.

**We may NOT claim:**
- **"First ZO training of quantized models"** — QuZO, QZO, ZeroQAT, HQZO, ZOQO already train/fine-tune quantized models with ZO (fine-tuning setting). ZeroQAT even names QuZO and QZO as the prior weight-only line.
- **"First ZO fine-tuning of quantized LLMs"** — that is QuZO's territory (EMNLP 2025).
- **"ZO avoids STE" as a novel insight** — QuZO's abstract states it and builds the method on it; ZeroQAT states it, adds a theoretical STE-bias analysis, and notes the ZO estimator is unbiased for a *smoothed* quantized objective (which also anticipates part of our smoothed-STE motivation).
- **"First gradient-free training of quantized networks"** — GFT (arXiv:2410.09734) is gradient-free training of quantized networks, from scratch, albeit small vision models and discrete search rather than population-based ZO.
- **"First quantized pretraining from scratch"** — BitNet b1.58/a4.8/v2 do it with backprop+STE; FOGZO does quantization-aware pre-training (with ZO augmentation of STE).
- **"First ZO/ES pretraining of language models"** — Allaire et al. 2025, KronZO 2026, SOMA 2026, Dust 2026.
- **Novelty of activation/node perturbation itself** — Widrow & Lehr 1990, Werfel et al. 2003, GEMINI 1988, Ren et al. 2023, Zoop 2025, Dust 2026.
- Any general superiority of "ZO over STE": FOGZO shows STE+ZO hybrids can beat pure STE in QAT pretraining; our Gaussian-smoothed STE control exists precisely to separate "zeroth-order-ness" from "smoothing" effects, so attribute effects carefully and only at our scales.
- Any efficiency or frontier-scaling claims: Dust itself disclaims compute efficiency; our small-scale results license none.
- Do not describe ZeroQAT's "quantized pre-training" as from-scratch — those experiments continue training *pretrained* Llama/OPT models under quantization; misdescribing it invites correction.

---

## 11. Overlap risks

1. **FOGZO (NeurIPS 2025, arXiv:2510.23926)** — highest risk on the (from-scratch × quantized × ZO-adjacent) axes: quantization-aware pre-training with an STE+ZO hybrid. Differentiators to state explicitly: it perturbs weights, requires backprop for its first-order guidance (not backprop-free), and keeps the STE; we are pure ZO, activation-space, STE-free. Must be cited and differentiated in the intro, not buried.
2. **EGGROLL's integer-datatype pretraining (arXiv:2511.16652)** — the abstract reports "stable pretraining of nonlinear recurrent language models that operate purely in integer datatypes." If that experiment is quantized-weight from-scratch training under ES, it weakens even our hedged "first ZO quantized from-scratch" claim (though it remains weight-space and non-transformer). **Action: read that section in full before submission.**
3. **GFT (arXiv:2410.09734)** — gradient-free from-scratch training of quantized networks (small vision models, discrete subset updates). Differentiate on: language-model pretraining, transformer scale, activation-space population estimator, and matched STE baselines.
4. **ZeroQAT (arXiv:2509.00031)** — closest in wording ("zeroth-order QAT," "quantized pre-training," weight+activation quantization, STE-free). Reviewers may conflate it with ours. Differentiate: it trains *pretrained* LLMs (continued training/fine-tuning) with weight perturbation; we train from scratch with activation perturbation.
5. **QZO-FF (NeurIPS 2024, arXiv:2411.04036)** — quantized (fixed-point) forward-gradient training on edge devices; weight-space, adaptation not pretraining, but shares "quantized forward + no backprop."
6. **CAQ-ZO (arXiv:2605.10673)** — a growing 2026 theory line on ZO through quantized forwards; a pretraining extension would land near our cell.
7. **AGZO / ZO-Act (2026)** — activation-*informed* weight-subspace ZO fine-tuning. Name-collision risk with "activation perturbation"; disambiguate explicitly. **[AGZO primary source unverified; ZO-Act authors not captured in this pass — re-verify before citing.]**
8. **Q Labs follow-ups to Dust** — none found as of 2026-10-11 (paper ≈1 week old; HN/press coverage exists). Dust's conclusion lists compute efficiency and non-differentiable architectures as future work; quantization is an obvious next step for them or any group. **Time-sensitive risk — re-run citation searches on Dust before submission.**
9. **The accelerating ZO-pretraining line** (Allaire/KronZO group; SOMA by Chaubard, Kochenderfer, Ré) — any of these adding quantization would directly enter our cell; monitor arXiv listings for "zeroth-order pretraining" + quantization.
10. **Hessian-ZO line** (HiZOO → HQZO) — already reached quantized fine-tuning in a Springer LNCS venue; a pretraining extension is plausible.

**Standing verification notes:** HQZO verified via Crossref (Springer page unrenderable for our fetcher) — abstract read only in part; Zoop verified via Dust's bibliography only (OpenReview bot-blocked); EGGROLL's integer-datatype experiment verified at abstract level only. All other references above were verified by reading their abstract page or full text on 2026-10-11.
