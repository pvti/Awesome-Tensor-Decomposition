# [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) Tensor Decomposition 
A curated list of tensor decomposition resources for *model compression*.

## :clipboard: Research Papers
### Transformer, LLM & more
| Title                                                        | Venue      | Year |
| ------------------------------------------------------------ | ---------- | ---- |
| [LS-PRISM: A layer-selective pruning method via low-rank approximation and sparsification for efficient large language model compression](https://doi.org/10.1016/j.neunet.2025.107909) | Neural Networks | 2025 |
| [Beyond Low-rank Decomposition: A Shortcut Approach for Efficient On-Device Learning](https://arxiv.org/abs/2505.05086) | ICML | 2025 |
| [Canonical Rank Adaptation: An Efficient Fine-Tuning Strategy for Vision Transformers](https://icml.cc/virtual/2025/poster/43662) | ICML | 2025 |
| [FLAT-LLM: Fine-grained Low-rank Activation Space Transformation for Large Language Model Compression](https://arxiv.org/abs/2505.23966) [![GitHub Repo stars](https://img.shields.io/github/stars/TTTTTTris/FLAT-LLM)](https://github.com/TTTTTTris/FLAT-LLM)| Arxiv | 2025 |
| [SoLA: Leveraging Soft Activation Sparsity and Low-Rank Decomposition for Large Language Model Compression](https://ojs.aaai.org/index.php/AAAI/article/view/33923) [![GitHub Repo stars](https://img.shields.io/github/stars/Ppaddington/SoLA)](https://github.com/Ppaddington/SoLA)| AAAI | 2025 |
| [SVD-LLM V2: Optimizing Singular Value Truncation for Large Language Model Compression](https://arxiv.org/abs/2503.12340) [![GitHub Repo stars](https://img.shields.io/github/stars/AIoT-MLSys-Lab/SVD-LLM)](https://github.com/AIoT-MLSys-Lab/SVD-LLM)| NAACL | 2025 |
| [NdLinear Is All You Need for Representation Learning](https://arxiv.org/abs/2503.17353) [![GitHub Repo stars](https://img.shields.io/github/stars/ensemble-core/NdLinear)](https://github.com/ensemble-core/NdLinear)| Arxiv | 2025 |
| [xKV: Cross-Layer SVD for KV-Cache Compression](https://arxiv.org/abs/2503.18893) [![GitHub Repo stars](https://img.shields.io/github/stars/abdelfattah-lab/xKV)](https://github.com/abdelfattah-lab/xKV)| Arxiv | 2025 |
| [You Only Prune Once: Designing Calibration-Free Model Compression With Policy Learning](https://arxiv.org/abs/2501.15296) [![GitHub Repo stars](https://img.shields.io/github/stars/LCS2-IIITD/PruneNet)](https://github.com/LCS2-IIITD/PruneNet)| ICLR | 2025 |
| [SliceGPT: Compress Large Language Models by Deleting Rows and Columns](https://arxiv.org/abs/2401.15024) [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/TransformerCompression)](https://github.com/microsoft/TransformerCompression)| ICLR | 2025 |
| [Efficient Learning With Sine-Activated Low-rank Matrices](https://arxiv.org/abs/2403.19243)| ICLR | 2025 |
| [Palu: Compression KV-Cache with Low-Rank Decomposition](https://arxiv.org/abs/2407.21118) [![GitHub Repo stars](https://img.shields.io/github/stars/shadowpa0327/Palu)](https://github.com/shadowpa0327/Palu)| ICLR | 2025 |
| [Dobi-SVD: Differentiable SVD for LLM Compression and Some New Perspectives](https://arxiv.org/abs/2502.02723) [![GitHub Repo stars](https://img.shields.io/github/stars/wangqinsi1/Dobi-SVD)](https://github.com/wangqinsi1/Dobi-SVD)| ICLR | 2025 |
| [MoDeGPT: Modular Decomposition for Large Language Model Compression](https://arxiv.org/abs/2408.09632) | ICLR | 2025 |
| [Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression](https://arxiv.org/abs/2410.03765) [![GitHub Repo stars](https://img.shields.io/github/stars/TUDa-HWAI/Basis_Sharing)](https://github.com/TUDa-HWAI/Basis_Sharing)| ICLR | 2025 |
| [SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression](https://arxiv.org/abs/2403.07378) [![GitHub Repo stars](https://img.shields.io/github/stars/AIoT-MLSys-Lab/SVD-LLM)](https://github.com/AIoT-MLSys-Lab/SVD-LLM)| ICLR | 2025 |
| [HASSLE-free: A unified Framework for Sparse plus Low-Rank Matrix Decomposition for LLMs](https://arxiv.org/abs/2502.00899) [![GitHub Repo stars](https://img.shields.io/github/stars/mazumder-lab/HASSLE-free)](https://github.com/mazumder-lab/HASSLE-free)| CPAL | 2025 |
| [FLAR-SVD: Fast and Latency-Aware Singular Value Decomposition for Model Compression](https://openaccess.thecvf.com/content/CVPR2025W/MAI/papers/Thoma_FLAR-SVD_Fast_and_Latency-Aware_Singular_Value_Decomposition_for_Model_Compression_CVPRW_2025_paper.pdf) [![GitHub Repo stars](https://img.shields.io/github/stars/MoritzTho/FLAR-SVD)](https://github.com/MoritzTho/FLAR-SVD)| CVPRW | 2025 |
| [Breaking the Low-Rank Dilemma of Linear Attention](https://arxiv.org/abs/2411.07635) [![GitHub Repo stars](https://img.shields.io/github/stars/qhfan/RALA)](https://github.com/qhfan/RALA)| CVPR | 2025 |
| [FlexiGPT: Pruning and Extending Large Language Models with Low-Rank Weight Sharing](https://arxiv.org/abs/2501.14713) | NAACL | 2025 |
| [TensorLLM: Tensorising Multi-Head Attention for Enhanced Reasoning and Compression in LLMs](https://arxiv.org/abs/2501.15674) | Arxiv | 2025 |
| [Memory-Efficient Vision Transformers: An Activation-Aware Mixed-Rank Compression Strategy](https://arxiv.org/abs/2402.06004) | ECCVW | 2024 |
| [Searching for Efficient Linear Layers over a Continuous Space of Structured Matrices](https://arxiv.org/abs/2410.02117) [![GitHub Repo stars](https://img.shields.io/github/stars/AndPotap/einsum-search)](https://github.com/AndPotap/einsum-search)| NeurIPS | 2024 |
| [DBA: Efficient Transformer With Dynamic Bilinear Low-Rank Attention](https://ieeexplore.ieee.org/document/10843139) | TNNLS | 2024 |
| [Tensor Product Attention Is All You Need](https://arxiv.org/abs/2501.06425) [![GitHub Repo stars](https://img.shields.io/github/stars/tensorgi/T6)](https://github.com/tensorgi/T6)| Arxiv | 2025 |
| [EoRA: Training-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation](https://arxiv.org/abs/2410.21271) | Arxiv | 2024 |
| [Compressing Large Language Models using Low Rank and Low Precision Decomposition](https://arxiv.org/abs/2405.18886) [![GitHub Repo stars](https://img.shields.io/github/stars/pilancilab/caldera)](https://github.com/pilancilab/caldera)| NeurIPS | 2024 |
| [ESPACE: Dimensionality Reduction of Activations for Model Compression](https://openreview.net/forum?id=HAcaANQNMK) [![GitHub Repo stars](https://img.shields.io/github/stars/NVlabs/ESPACE)](https://github.com/NVlabs/ESPACE)| NeurIPS | 2024 |
| [MoE-I2: Compressing Mixture of Experts Models through Inter-Expert Pruning and Intra-Expert Low-Rank Decomposition](https://arxiv.org/abs/2411.01016) [![GitHub Repo stars](https://img.shields.io/github/stars/xiaochengsky/MoEI-2)](https://github.com/xiaochengsky/MoEI-2)| EMNLP 2024 Findings | 2024 |
| [Tensor Train Low-rank Approximation (TT-LoRA): Democratizing AI with Accelerated LLMs](https://arxiv.org/abs/2408.01008) | Arxiv | 2024 |
| [Dual-grained Lightweight Strategy](https://ieeexplore.ieee.org/document/10713325) | TPAMI | 2024 |
| [Adaptive Feature-based Low-Rank Compression of Large Language Models via Bayesian Optimization](https://arxiv.org/abs/2405.10616) | EMNLP findings | 2024 |
| [Geometry is All You Need: A Unified Taxonomy of Matrix and Tensor Factorization for Compression of Generative Language Models](https://arxiv.org/abs/2410.03040) | Arxiv | 2024 |
| [From GaLore to WeLore: How Low-Rank Weights Non-uniformly Emerge from Low-Rank Gradients](https://arxiv.org/abs/2407.11239) [![GitHub Repo stars](https://img.shields.io/github/stars/VITA-Group/WeLore)](https://github.com/VITA-Group/WeLore)| Arxiv | 2024 |
| [The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction](https://arxiv.org/abs/2312.13558) [![GitHub Repo stars](https://img.shields.io/github/stars/pratyushasharma/laser)](https://github.com/pratyushasharma/laser)| ICLR | 2024 |
| [ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models](https://arxiv.org/abs/2312.05821) [![GitHub Repo stars](https://img.shields.io/github/stars/hahnyuan/ASVD4LLM)](https://github.com/hahnyuan/ASVD4LLM)| Arxiv | 2024 |
| [AdaZeta: Adaptive Zeroth-Order Tensor-Train Adaption for Memory-Efficient Large Language Models Fine-Tuning](https://arxiv.org/abs/2406.18060) [![GitHub Repo stars](https://img.shields.io/github/stars/yifanycc/AdaZeta)](https://github.com/yifanycc/AdaZeta)| EMNLP | 2024 |
| [Adaptive Rank Selections for Low-Rank Approximation of Language Models](https://aclanthology.org/2024.naacl-long.13/) [![GitHub Repo stars](https://img.shields.io/github/stars/sidhantls/adaptive-rank-selection-svd)](https://github.com/sidhantls/adaptive-rank-selection-svd)| NAACL | 2024 |
| [Dynamic Low-rank Estimation for Transformer-based Language Models](https://aclanthology.org/2023.findings-emnlp.621/) | NAACL Findings | 2024 |
| [PELA: Learning Parameter-Efficient Models with Low-Rank Approximation](https://arxiv.org/abs/2310.10700) [![GitHub Repo stars](https://img.shields.io/github/stars/guoyang9/PELA)](https://github.com/guoyang9/PELA)| CVPR | 2024 |
| [Adaptive Rank Selections for Low-Rank Approximation of Language Models](https://aclanthology.org/2024.naacl-long.13/)| ACL | 2024 |
| [TRAWL: Tensor Reduced and Approximated Weights for Large Language Models](https://arxiv.org/abs/2406.17261) | Arxiv | 2024 |
| [Tender: Accelerating Large Language Models via Tensor Decomposition and Runtime Requantization](https://arxiv.org/abs/2406.12930)  [![GitHub Repo stars](https://img.shields.io/github/stars/snu-comparch/Tender)](https://github.com/snu-comparch/Tender)| ISCA | 2024 |
| [Unified Low-rank Compression Framework for Click-through Rate Prediction](https://arxiv.org/abs/2405.18146) [![GitHub Repo stars](https://img.shields.io/github/stars/yuhao318/atomic_feature_mimicking)](https://github.com/yuhao318/atomic_feature_mimicking)| KDD2024 ADS | 2024 |
| [LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models](https://arxiv.org/abs/2402.11417) [![GitHub Repo stars](https://img.shields.io/github/stars/yifanycc/loretta)](https://github.com/yifanycc/loretta)| NAACL | 2024 |
| [CoMERA: Computing- and Memory-Efficient Training via Rank-Adaptive Tensor Optimization](https://arxiv.org/abs/2405.14377) | Arxiv | 2024 |
| [FLORA: Fine-grained Low-Rank Architecture Search for Vision Transformer](https://arxiv.org/abs/2311.03912) [![GitHub Repo stars](https://img.shields.io/github/stars/shadowpa0327/FLORA)](https://github.com/shadowpa0327/FLORA)| WACV | 2024 |
| [FLoRA: Low-Rank Core Space for N-dimension](https://arxiv.org/abs/2405.14739) [![GitHub Repo stars](https://img.shields.io/github/stars/SJTU-DeepVisionLab/FLoRA)](https://github.com/SJTU-DeepVisionLab/FLoRA)| Arxiv | 2024 |
| [Enhancing GAN Performance Through Neural Architecture Search and Tensor Decomposition](https://ieeexplore.ieee.org/document/10446488) [![GitHub Repo stars](https://img.shields.io/github/stars/PrasannaPulakurthi/MMD-AdversarialNAS)](https://github.com/PrasannaPulakurthi/MMD-AdversarialNAS)| ICASSP | 2024 |
| [A Computational Study of Matrix Decomposition Methods for Compression of Pre-trained Transformers](https://aclanthology.org/2023.paclic-1.73/) [![GitHub Repo stars](https://img.shields.io/github/stars/s-nlp/compression)](https://github.com/s-nlp/compression)| PACLIC | 2024 |
| [Singularformer: Learning to Decompose Self-Attention to Linearize the Complexity of Transformer](https://www.ijcai.org/proceedings/2023/493) [![GitHub Repo stars](https://img.shields.io/github/stars/CSUBioGroup/Singularformer)](https://github.com/CSUBioGroup/Singularformer)| IJCAI | 2023 |
| [LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation](https://arxiv.org/abs/2306.11222) [![GitHub Repo stars](https://img.shields.io/github/stars/yxli2123/LoSparse)](https://github.com/yxli2123/LoSparse)| ICML | 2023 |
| [Singularformer: Learning to Decompose Self-Attention to Linearize the Complexity of Transformer](https://www.ijcai.org/proceedings/2023/493) [![GitHub Repo stars](https://img.shields.io/github/stars/CSUBioGroup/Singularformer)](https://github.com/CSUBioGroup/Singularformer)| IJCAI | 2023 |
| [FacT: Factor-Tuning for Lightweight Adaptation on Vision Transformer](https://arxiv.org/abs/2212.03145) [![GitHub Repo stars](https://img.shields.io/github/stars/JieShibo/PETL-ViT)](https://github.com/JieShibo/PETL-ViT)| AAAI | 2023 |
| [LightFormer: Light-weight Transformer Using SVD-based Weight Transfer and Parameter Sharing](https://aclanthology.org/2023.findings-acl.656/) | ACL | 2023 |
| [Compressing Transformers: Features Are Low-Rank, but Weights Are Not!](https://ojs.aaai.org/index.php/AAAI/article/view/26304) | AAAI | 2023 |
| [TT-MLP: Tensor Train Decomposition on Deep MLPs](https://ieeexplore.ieee.org/document/10032168) | IEEE Access | 2023 |
| [Matrix Compression via Randomized Low Rank and Low Precision Factorization](https://arxiv.org/abs/2310.11028) [![GitHub Repo stars](https://img.shields.io/github/stars/pilancilab/matrix-compressor)](https://github.com/pilancilab/matrix-compressor)| NeurIPS | 2023 |
| [Strategies for Applying Low Rank Decomposition to Transformer-Based Models](https://neurips2022-enlsp.github.io/papers/paper_33.pdf) | NeurIPS | 2022 |
| [Hypoformer: Hybrid Decomposition Transformer for Edge-friendly Neural Machine Translation](https://aclanthology.org/2022.emnlp-main.475/) | EMNLP | 2022 |
| [Kronecker Decomposition for GPT Compression](https://aclanthology.org/2022.acl-short.24/) | ACL | 2022 |
| [Deeply Tensor Compressed Transformers for End-to-End Object Detection](https://ojs.aaai.org/index.php/AAAI/article/view/20397) | AAAI | 2022 |
| [Language model compression with weighted low-rank factorization](https://arxiv.org/abs/2207.00112) [![GitHub Repo stars](https://img.shields.io/github/stars/RahulSChand/Weighted-low-rank-factorization-Pytorch)](https://github.com/RahulSChand/Weighted-low-rank-factorization-Pytorch)| ICLR | 2022 |
| [DRONE: Data-aware Low-rank Compression for Large NLP Models](https://proceedings.neurips.cc/paper/2021/hash/f56de5ef149cf0aedcc8f4797031e229-Abstract.html) | NeurIPS | 2021 |
| [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768) [![GitHub Repo stars](https://img.shields.io/github/stars/lucidrains/linformer)](https://github.com/lucidrains/linformer)| Arxiv | 2020 |
| [Lightweight and Efficient End-to-End Speech Recognition Using Low-Rank Transformer](https://arxiv.org/abs/1910.13923) [![GitHub Repo stars](https://img.shields.io/github/stars/gentaiscool/end2end-asr-pytorch)](https://github.com/gentaiscool/end2end-asr-pytorch) [![GitHub Repo stars](https://img.shields.io/github/stars/SamuelCahyawijaya/greenformer)](https://github.com/SamuelCahyawijaya/greenformer)| ICASSP | 2020 |
| [A Tensorized Transformer for Language Modeling](https://arxiv.org/abs/1906.09777) [![GitHub Repo stars](https://img.shields.io/github/stars/szhangtju/The-compression-of-Transformer)](https://github.com/szhangtju/The-compression-of-Transformer)| NeurIPS | 2019 |
| [GroupReduce: Block-Wise Low-Rank Approximation for Neural Language Model Shrinking](https://arxiv.org/abs/1806.06950)| NIPS | 2018 |

### CNN
| Title                                                        | Venue      | Year |
| ------------------------------------------------------------ | ---------- | ---- |
| [An overview of tensor and matrix decomposition methods, applied to deep neural networks](https://link.springer.com/article/10.1007/s12530-025-09733-6) | Evolving Systems | 2025 |
| [FETTA: Flexible and Efficient Hardware Accelerator for Tensorized Neural Network Training](https://arxiv.org/abs/2504.06474) | Arxiv | 2025 |
| [Fine-grained hierarchical singular value decomposition for convolutional neural networks compression and acceleration](https://doi.org/10.1016/j.neucom.2025.129966) | Neurocomputing | 2025 |
| [Low-Rank Compression for IMC Arrays](https://www.arxiv.org/abs/2502.07820) | Arxiv | 2025 |
| [Enhancing GANs With MMD Neural Architecture Search, PMish Activation Function, and Adaptive Rank Decomposition](https://ieeexplore.ieee.org/document/10732016) [![GitHub Repo stars](https://img.shields.io/github/stars/PrasannaPulakurthi/MMD-PMish-NAS-GAN)](https://github.com/PrasannaPulakurthi/MMD-PMish-NAS-GAN) | IEEE ACCESS | 2024 |
| [Quantization Aware Factorization for Deep Neural Network Compression](https://www.jair.org/index.php/jair/article/view/16167) | JAIR | 2024 |
| [J-Tucker: Joint Compression Scheme for Efficient Deployment of Multi-Task Deep Learning Models on Edge Devices](https://ieeexplore.ieee.org/document/10794779) | IEEE Network | 2024 |
| [3D-KCPNet: Efficient 3DCNNs based on tensor mapping theory](https://doi.org/10.1016/j.neucom.2023.126985) | Neurocomputing | 2024 |
| [Fully Tensorized Lightweight ConvLSTM Neural Networks for Hyperspectral Image Classification](https://ieeexplore.ieee.org/abstract/document/10798991)| TNNLS | 2024 |
| [TEC-CNN: Towards Efficient Compressing Convolutional Neural Nets with Low-rank Tensor Decomposition](https://dl.acm.org/doi/10.1145/3702641)| TOMM | 2024 |
| [Activation Map Compression through Tensor Decomposition for Deep Learning](https://neurips.cc/virtual/2024/poster/95125) [![GitHub Repo stars](https://img.shields.io/github/stars/Le-TrungNguyen/NeurIPS2024-ActivationCompression)](https://github.com/Le-TrungNguyen/NeurIPS2024-ActivationCompression)| NeurIPS | 2024 |
| [Geometry-aware training of factorized layers in tensor Tucker format](https://arxiv.org/abs/2305.19059)| NeurIPS | 2024 |
| [Robustness of Tensor Decomposition-Based Neural Network Compression](https://ieeexplore.ieee.org/document/10647942)| ICIP | 2024 |
| [How to Train Your Unstable Looped Tensor Network](https://ieeexplore.ieee.org/document/10695808)| JSTSP | 2024 |
| [Learning Low-Rank Tensor Cores with Probabilistic l0-Regularized Rank Selection for Model Compression](https://www.ijcai.org/proceedings/2024/418) [![GitHub Repo stars](https://img.shields.io/github/stars/ctxGou/Tensor-L0-Compression)](https://github.com/ctxGou/Tensor-L0-Compression)| IJCAI | 2024 |
| [Structure-Preserving Network Compression Via Low-Rank Induced Training Through Linear Layers Composition](https://arxiv.org/abs/2405.03089) [![GitHub Repo stars](https://img.shields.io/github/stars/XitongSystem/LoRITa)](https://github.com/XitongSystem/LoRITa)| TMLR | 2024 |
| [Compact Model Training by Low-Rank Projection With Energy Transfer](https://ieeexplore.ieee.org/document/10551437) [![GitHub Repo stars](https://img.shields.io/github/stars/BZQLin/LRPET)](https://github.com/BZQLin/LRPET)| TNNLS | 2024 |
| [An Accuracy-Preserving Neural Network Compression via Tucker Decomposition](https://ieeexplore.ieee.org/document/10614384)| IEEE Transactions on Sustainable Computing | 2024 |
| [Convolution Filter Compression via Sparse Linear Combinations of Quantized Basis](https://ieeexplore.ieee.org/document/10689618)| TNNLS | 2024 |
| [Co-Exploring Structured Sparsification and Low-Rank Tensor Decomposition for Compact DNNs](https://ieeexplore.ieee.org/document/10574865)| TNNLS | 2024 |
| [Coarse-To-Fine Tensor Trains for Compact Visual Representations](https://arxiv.org/abs/2406.04332) [![GitHub Repo stars](https://img.shields.io/github/stars/sebulo/PuTT)](https://github.com/sebulo/PuTT)| ICML | 2024 |
| [Position: Tensor Networks are a Valuable Asset for Green AI](https://openreview.net/pdf?id=mcg6jppkwb)| ICML | 2024 |
| [Compression-aware Training of Neural Networks using Frank-Wolfe](https://arxiv.org/abs/2205.11921) [![GitHub Repo stars](https://img.shields.io/github/stars/ZIB-IOL/compression-aware-SFW)](https://github.com/ZIB-IOL/compression-aware-SFW)| Arxiv | 2024 |
| [A Practical Approach for Employing Tensor Train Decomposition in Edge Devices](https://link.springer.com/article/10.1007/s10766-024-00762-3) | IJPP | 2024 |
| [Structure-Preserving Network Compression Via Low-Rank Induced Training Through Linear Layers Composition](https://arxiv.org/abs/2405.03089) | Arxiv | 2024 |
| [Reduced storage direct tensor ring decomposition for convolutional neural networks compression](https://arxiv.org/abs/2405.10802) [![GitHub Repo stars](https://img.shields.io/github/stars/mateuszgabor/rsdtr_compression)](https://github.com/mateuszgabor/rsdtr_compression)| Arxiv | 2024 |
| [Federated Learning Using Coupled Tensor Train Decomposition](https://arxiv.org/abs/2403.02898) | Arxiv | 2024 |
| [Neural Network Compression Based on Tensor Ring Decomposition](https://ieeexplore.ieee.org/abstract/document/10510501) | TNNLS | 2024 |
| [Enhanced network compression through tensor decompositions and pruning](https://hal.science/hal-04475167v1) [![GitHub Repo stars](https://img.shields.io/github/stars/pvti/NORTON)](https://github.com/pvti/NORTON)| TNNLS | 2024 |
| [Deep Convolutional Neural Network Compression Method: Tensor Ring Decomposition with Variational Bayesian Approach](https://link.springer.com/article/10.1007/s11063-024-11465-8) | Neural Processing Letters | 2024 |
| [Deep Learning Model Compression With Rank Reduction in Tensor Decomposition](https://ieeexplore.ieee.org/abstract/document/10321737) | TNNLS | 2023 |
| [MARS: Masked Automatic Ranks Selection in Tensor Decompositions](https://arxiv.org/abs/2006.10859) [![GitHub Repo stars](https://img.shields.io/github/stars/MaxBourdon/mars)](https://github.com/MaxBourdon/mars)| AISTATS | 2023 |
| [Mixed-TD: Efficient Neural Network Accelerator with Layer-Specific Tensor Decomposition](https://arxiv.org/abs/2306.05021) [![GitHub Repo stars](https://img.shields.io/github/stars/Yu-Zhewen/Mixed-TD)](https://github.com/Yu-Zhewen/Mixed-TD)| FPL | 2023 |
| [SVD-NAS: Coupling Low-Rank Approximation and Neural Architecture Search](https://arxiv.org/abs/2208.10404) [![GitHub Repo stars](https://img.shields.io/github/stars/Yu-Zhewen/SVD-NAS)](https://github.com/Yu-Zhewen/SVD-NAS)| WACV | 2023 |
| [How Informative is the Approximation Error from Tensor Decomposition for Neural Network Compression?](https://arxiv.org/abs/2305.05318) | ICLR | 2023 |
| [Learning Low-Rank Representations for Model Compression](https://arxiv.org/abs/2211.11397) | IJCNN | 2023 |
| [Tensor shape search for efficient compression of tensorized data and neural networks](https://www.sciencedirect.com/science/article/pii/S1568494623010050) | Applied Soft Computing | 2023 |
| [Compressing convolutional neural networks with hierarchical Tucker-2 decomposition](https://www.sciencedirect.com/science/article/pii/S156849462200905X) [![GitHub Repo stars](https://img.shields.io/github/stars/mateuszgabor/ht2)](https://github.com/mateuszgabor/ht2)| Applied Soft Computing | 2023 |
| [Tensor shape search for efficient compression of tensorized data and neural networks](https://www.sciencedirect.com/science/article/pii/S1568494623010050) | Applied Soft Computing | 2023 |
| [An effective low-rank compression with a joint rank selection followed by a compression-friendly training](https://www.sciencedirect.com/science/article/pii/S0893608023000242) | Neural Networks | 2023 |
| [Joint matrix decomposition for deep convolutional neural networks compression](https://www.sciencedirect.com/science/article/pii/S0925231222012887) [![GitHub Repo stars](https://img.shields.io/github/stars/ShaowuChen/JointSVD)](https://github.com/ShaowuChen/JointSVD)| Neurocomputing | 2023 |
| [Training Acceleration of Low-Rank Decomposed Networks using Sequential Freezing and Rank Quantization](https://arxiv.org/abs/2309.03824) | Arxiv | 2023 |
| [Knowledge Transfer via Decomposing Essential Information in Convolutional Neural Networks](https://ieeexplore.ieee.org/document/9222552) [![GitHub Repo stars](https://img.shields.io/github/stars/sseung0703/KD_methods_with_TF)](https://github.com/sseung0703/KD_methods_with_TF)| TNNLS | 2022 |
| [Compression of Deep Neural Networks based on quantized tensor decomposition to implement on reconfigurable hardware platforms](https://www.sciencedirect.com/science/article/pii/S089360802200065X) | Neural Networks | 2022 |
| [Teacher–student knowledge distillation based on decomposed deep feature representation for intelligent mobile applications](https://doi.org/10.1016/j.eswa.2022.117474) | Expert Systems with Applications | 2022 |
| [HODEC: Towards Efficient High-Order DEcomposed Convolutional Neural Networks](https://ieeexplore.ieee.org/document/9879408) | CVPR | 2022 |
| [Towards Practical Control of Singular Values of Convolutional Layers](https://arxiv.org/abs/2211.13771) [![GitHub Repo stars](https://img.shields.io/github/stars/WhiteTeaDragon/practical_svd_conv)](https://github.com/WhiteTeaDragon/practical_svd_conv)| NeurIPS | 2022 |
| [Low-rank lottery tickets: finding efficient low-rank neural networks via matrix differential equations](https://arxiv.org/abs/2205.13571) [![GitHub Repo stars](https://img.shields.io/github/stars/COMPiLELab/DLRT-Net)](https://github.com/COMPiLELab/DLRT-Net)| NeurIPS | 2022 |
| [Realistic acceleration of neural networks with fine-grained tensor decomposition](https://doi.org/10.1016/j.neucom.2022.09.057) | Neurocomputing | 2022 |
| [BATUDE: Budget-Aware Neural Network Compression Based on Tucker Decomposition](https://ojs.aaai.org/index.php/AAAI/article/view/20869) | AAAI | 2022 |
| [Convolutional Neural Network Compression through Generalized Kronecker Product Decomposition](https://ojs.aaai.org/index.php/AAAI/article/view/19958) [![GitHub Repo stars](https://img.shields.io/github/stars/marawangamal/generalized-kronecker-product-decomposition)](https://github.com/marawangamal/generalized-kronecker-product-decomposition)| AAAI | 2022 |
| [Towards Compact Neural Networks via End-to-End Training: A Bayesian Tensor Approach with Automatic Rank Determination](https://arxiv.org/abs/2010.08689) [![GitHub Repo stars](https://img.shields.io/github/stars/colehawkins/bayesian-tensor-rank-determination)](https://github.com/colehawkins/bayesian-tensor-rank-determination)| SIMODS | 2022 |
| [A novel compact design of convolutional layers with spatial transformation towards lower-rank representation for image classification](https://doi.org/10.1016/j.knosys.2022.109723) [![GitHub Repo stars](https://img.shields.io/github/stars/liubc17/low_rank_compact_transformed)](https://github.com/liubc17/low_rank_compact_transformed)| Knowledge-Based Systems | 2022 |
| [Deep neural network compression by Tucker decomposition with nonlinear response](https://www.sciencedirect.com/science/article/pii/S0950705122000326) | Knowledge-Based Systems | 2022 |
| [Nested compression of convolutional neural networks with Tucker-2 decomposition](https://ieeexplore.ieee.org/abstract/document/9892959) | IJCNN | 2022 |
| [PSM-nets: Compressing Neural Networks with Product of Sparse Matrices](https://ieeexplore.ieee.org/abstract/document/9533408) | IJCNN | 2022 |
| [A Design Space Exploration Methodology for Enabling Tensor Train Decomposition in Edge Devices](https://link.springer.com/chapter/10.1007/978-3-031-15074-6_11) | SAMOS | 2022 |
| [Spatiotemporal Joint Filter Decomposition in 3D Convolutional Neural Networks](https://openreview.net/forum?id=Alr5_kKmLBX) | NeurIPS | 2021 |
| [Compressing Neural Networks: Towards Determining the Optimal Layer-wise Decomposition](https://arxiv.org/abs/2107.11442) [![GitHub Repo stars](https://img.shields.io/github/stars/lucaslie/torchprune)](https://github.com/lucaslie/torchprune)| NeurIPS | 2021 |
| [Deeply Shared Filter Bases for Parameter-Efficient Convolutional Neural Networks](https://arxiv.org/abs/2006.05066) [![GitHub Repo stars](https://img.shields.io/github/stars/ssregibility/Net_RL2)](https://github.com/ssregibility/Net_RL2)| NeurIPS | 2021 |
| [Tensor Regression Networks](https://arxiv.org/abs/1707.08308) [![GitHub Repo stars](https://img.shields.io/github/stars/JeanKossaifi/tensorly-notebooks)](https://github.com/JeanKossaifi/tensorly-notebooks)| JMLR | 2021 |
| [Parameter Efficient Dynamic Convolution via Tensor Decomposition](https://www.bmvc2021-virtualconference.com/conference/papers/paper_1631.html) [![GitHub Repo stars](https://img.shields.io/github/stars/zejiangh/PEDConv)](https://github.com/zejiangh/PEDConv)| BMVC | 2021 |
| [Nonlinear tensor train format for deep neural network compression](https://doi.org/10.1016/j.neunet.2021.08.028) | Neural Networks | 2021 |
| [Towards Efficient Tensor Decomposition-Based DNN Model Compression with Optimization Framework](https://ieeexplore.ieee.org/document/9577813) | CVPR | 2021 |
| [Pufferfish: Communication-efficient Models At No Extra Cost](https://arxiv.org/abs/2103.03936) [![GitHub Repo stars](https://img.shields.io/github/stars/hwang595/Pufferfish)](https://github.com/hwang595/Pufferfish)| MLSys | 2021 |
| [Learning-based Tensor Decomposition with Adaptive Rank Penalty for CNNs Compression](https://ieeexplore.ieee.org/document/9565537) | MIPR | 2021 |
| [Deep Convolutional Neural Network Compression via Coupled Tensor Decomposition](https://ieeexplore.ieee.org/document/9261106) | JSTSP | 2021 |
| [Tensor Reordering for CNN Compression](https://arxiv.org/abs/2010.12110) | ICASSP | 2021 |
| [Block-term tensor neural networks](https://www.sciencedirect.com/science/article/pii/S0893608020302045)| Neural Networks | 2020 |
| [Stable Low-rank Tensor Decomposition for Compression of Convolutional Neural Network](https://arxiv.org/abs/2008.05441) | ECCV | 2020 |
| [Low-Rank Compression of Neural Nets: Learning the Rank of Each Layer](https://ieeexplore.ieee.org/document/9157223) [![GitHub Repo stars](https://img.shields.io/github/stars/UCMerced-ML/LC-model-compression)](https://github.com/UCMerced-ML/LC-model-compression)| CVPR | 2020 |
| [Few Sample Knowledge Distillation for Efficient Network Compression](https://arxiv.org/abs/1812.01839) [![GitHub Repo stars](https://img.shields.io/github/stars/LTH14/FSKD)](https://github.com/LTH14/FSKD)| CVPR | 2020 |
| [Factorized Higher-Order CNNs with an Application to Spatio-Temporal Emotion Estimation](https://arxiv.org/abs/1906.06196) | CVPR | 2020 |
| [Learning Low-rank Deep Neural Networks via Singular Vector Orthogonality Regularization and Singular Value Sparsification](https://arxiv.org/abs/2004.09031) [![GitHub Repo stars](https://img.shields.io/github/stars/yanghr/SVD_Prune_EDLCV)](https://github.com/yanghr/SVD_Prune_EDLCV)| CVPRW | 2020 |
| [T-Basis: a Compact Representation for Neural Networks](https://arxiv.org/abs/2007.06631) [![GitHub Repo stars](https://img.shields.io/github/stars/toshas/tbasis)](https://github.com/toshas/tbasis)| ICML | 2020 |
| [PENNI: Pruned Kernel Sharing for Efficient CNN Inference](https://arxiv.org/abs/2005.07133) [![GitHub Repo stars](https://img.shields.io/github/stars/timlee0212/PENNI)](https://github.com/timlee0212/PENNI)| ICML | 2020 |
| [A Novel Rank Selection Scheme in Tensor Ring Decomposition Based on Reinforcement Learning for Deep Neural Networks](https://ieeexplore.ieee.org/document/9053292)| ICASSP | 2020 |
| [Holistic CNN Compression via Low-Rank Decomposition with Knowledge Transfer](https://ieeexplore.ieee.org/document/8478366) [![GitHub Repo stars](https://img.shields.io/github/stars/ShaohuiLin/LRDKT)](https://github.com/ShaohuiLin/LRDKT)| TPAMI | 2019 |
| [LTNN: A Layerwise Tensorized Compression of Multilayer Neural Network](https://ieeexplore.ieee.org/document/8480873)| TNNLS | 2019 |
| [Einconv: Exploring Unexplored Tensor Network Decompositions for Convolutional Neural Networks](https://arxiv.org/abs/1908.04471) [![GitHub Repo stars](https://img.shields.io/github/stars/pfnet-research/einconv)](https://github.com/pfnet-research/einconv)| NeurIPS | 2019 |
| [Efficient Neural Network Compression](https://arxiv.org/abs/1811.12781) [![GitHub Repo stars](https://img.shields.io/github/stars/Hyeji-Kim/ENC)](https://github.com/Hyeji-Kim/ENC)| CVPR | 2019 |
| [ADA-Tucker: Compressing deep neural networks via adaptive dimension adjustment tucker decomposition](https://www.sciencedirect.com/science/article/pii/S0893608018303010) | Neural Networks | 2019 |
| [RotDCF: Decomposition of Convolutional Filters for Rotation-Equivariant Deep Networks](https://arxiv.org/abs/1805.06846) [![GitHub Repo stars](https://img.shields.io/github/stars/ZichenMiao/RotDCF)](https://github.com/ZichenMiao/RotDCF)| ICLR | 2019 |
| [Learning Filter Basis for Convolutional Neural Network Compression](https://arxiv.org/abs/1908.08932) [![GitHub Repo stars](https://img.shields.io/github/stars/ofsoundof/learning_filter_basis)](https://github.com/ofsoundof/learning_filter_basis)| ICCV | 2019 |
| [Automated Multi-Stage Compression of Neural Networks](https://ieeexplore.ieee.org/document/9022596) [![GitHub Repo stars](https://img.shields.io/github/stars/musco-ai/musco-pytorch)](https://github.com/musco-ai/musco-pytorch)| ICCVW | 2019 |
| [Compressing Deep Models using Multi Tensor Train Decomposition](https://ieeexplore.ieee.org/document/9074612) | ICCAIS | 2019 |
| [Compressing Fully Connected Layers using Kronecker Tensor Decomposition](https://ieeexplore.ieee.org/document/8962432) | ICCSNT | 2019 |
| [Adaptive Mixture of Low-Rank Factorizations for Compact Neural Modeling](https://openreview.net/forum?id=r1xFE3Rqt7) [![GitHub Repo stars](https://img.shields.io/github/stars/zuenko/ALRF)](https://github.com/zuenko/ALRF)| OpenReview | 2019 |
| [Sharing Residual Units Through Collective Tensor Factorization To Improve Deep Neural Networks](https://arxiv.org/abs/1703.02180) | IJCAI | 2018 |
| [Learning Compact Recurrent Neural Networks with Block-Term Tensor Decomposition](https://arxiv.org/abs/1712.05134) | CVPR | 2018 |
| [Wide Compression: Tensor Ring Nets](https://arxiv.org/abs/1802.09052) | CVPR | 2018 |
| [DCFNet: Deep Neural Network with Decomposed Convolutional Filters](https://arxiv.org/abs/1802.04145) [![GitHub Repo stars](https://img.shields.io/github/stars/ZeWang95/DCFNet-Pytorch)](https://github.com/ZeWang95/DCFNet-Pytorch)| ICML | 2018 |
| [Self-supervised Knowledge Distillation Using Singular Value Decomposition](https://arxiv.org/abs/1807.06819) [![GitHub Repo stars](https://img.shields.io/github/stars/sseung0703/SSKD_SVD)](https://github.com/sseung0703/SSKD_SVD)| ECCV | 2018 |
| [Extreme Network Compression via Filter Group Approximation](https://arxiv.org/abs/1807.11254) | ECCV | 2018 |
| [Network Decoupling: From Regular to Depthwise Separable Convolutions](https://arxiv.org/abs/1808.05517) [![GitHub Repo stars](https://img.shields.io/github/stars/JianboGuo/network-decoupling)](https://github.com/JianboGuo/network-decoupling)| BMVC | 2018 |
| [On Compressing Deep Models by Low Rank and Sparse Decomposition](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_On_Compressing_Deep_CVPR_2017_paper.pdf) | CVPR | 2017 |
| [Coordinating Filters for Faster Deep Neural Networks](https://arxiv.org/abs/1703.09746) [![GitHub Repo stars](https://img.shields.io/github/stars/wenwei202/caffe)](https://github.com/wenwei202/caffe)| ICCV | 2017 |
| [Factorized Convolutional Neural Networks](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w10/Wang_Factorized_Convolutional_Neural_ICCV_2017_paper.pdf) | ICCVW | 2017 |
| [Tensor Regression Networks with various Low-Rank Tensor Approximations](https://arxiv.org/abs/1712.09520) [![GitHub Repo stars](https://img.shields.io/github/stars/xwcao/LowRankTRN)](https://github.com/xwcao/LowRankTRN)| Arxiv | 2017 |
| [Accelerating Very Deep Convolutional Networks for Classification and Detection](https://arxiv.org/abs/1505.06798) | TPAMI | 2016 |
| [Convolutional Neural Networks With Low-rank Regularization](https://arxiv.org/abs/1511.06067) [![GitHub Repo stars](https://img.shields.io/github/stars/chengtaipu/lowrankcnn)](https://github.com/chengtaipu/lowrankcnn)| ICLR | 2016 |
| [Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications](https://arxiv.org/abs/1511.06530) | ICLR | 2016 |
| [Towards Convolutional Neural Networks Compression via Global Error Reconstruction](https://www.ijcai.org/Proceedings/16/Papers/251.pdf) | IJCAI | 2016 |
| [Accelerating Convolutional Neural Networks for Mobile Applications](https://doi.org/10.1145/2964284.2967280) | MM | 2016 |
| [Ultimate tensorization: compressing convolutional and FC layers alike](https://arxiv.org/abs/1611.03214) [![GitHub Repo stars](https://img.shields.io/github/stars/timgaripov/TensorNet-TF)](https://github.com/timgaripov/TensorNet-TF) | NIPSW | 2016 |
| [Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition](https://arxiv.org/abs/1412.6553) | ICLR | 2015 |
| [Speeding up Convolutional Neural Networks with Low Rank Expansions](https://arxiv.org/abs/1405.3866) | Arxiv | 2014 |

### RNN
| Title                                                        | Venue      | Year |
| ------------------------------------------------------------ | ---------- | ---- |
| [Kronecker CP Decomposition With Fast Multiplication for Compressing RNNs](https://arxiv.org/abs/2008.09342) | TNNLS | 2021 |

## :books: Surveys
| Title                                                        | Venue      | Year |
| ------------------------------------------------------------ | ---------- | ---- |
| [Tensor Decompositions for Signal Processing: Theory, Advances, and Applications](https://hal.science/hal-05135784/) | Signal Processing | 2025 |
| [Low Rank Optimization for Efficient Deep Learning: Making a Balance Between Compact Architecture And Fast Training](https://ieeexplore.ieee.org/document/10355073) | Journal of Systems Engineering and Electronics | 2024 |
| [Tensor Decomposition for Model Reduction in Neural Networks: A Review](https://arxiv.org/abs/2304.13539) | IEEE Circuits and Systems Magazine | 2023 |
| [Low Rank Optimization for Efficient Deep Learning: Making A Balance between Compact Architecture and Fast Training](https://arxiv.org/abs/2303.13635)|  Journal of Systems Engineering and Electronics | 2023 |
| [Tensor Networks Meet Neural Networks: A Survey and Future Perspectives](https://arxiv.org/abs/2302.09019) [![GitHub Repo stars](https://img.shields.io/github/stars/tnbar/awesome-tensorial-neural-networks)](https://github.com/tnbar/awesome-tensorial-neural-networks)| Arxiv | 2023 |
| [High-performance tensor decompositions for compressing and accelerating deep neural networks](https://www.sciencedirect.com/science/article/abs/pii/B9780128244470000157) [![GitHub Repo stars](https://img.shields.io/github/stars/YangletLiu/Tensor_Layer_for_Deep_Neural_Network_Compression)](https://github.com/YangletLiu/Tensor_Layer_for_Deep_Neural_Network_Compression)| Tensors for Data Processing | 2022 |
| [Tensor Methods in Computer Vision and Deep Learning](https://arxiv.org/abs/2107.03436) | Proceedings of the IEEE | 2021 |
| [Tensor Decomposition for Signal Processing and Machine Learning](https://ieeexplore.ieee.org/abstract/document/7891546) | IEEE Transactions on Signal Processing | 2017 |
| [A literature survey of low-rank tensor approximation techniques](https://arxiv.org/abs/1302.7121) | GAMM-Mitteilungen | 2013 |
| [The Higher-Order Singular Value Decomposition: Theory and an Application](https://ieeexplore.ieee.org/document/5447070) | IEEE Signal Processing Magazine | 2010 |
| [Tensor Decompositions and Applications](https://epubs.siam.org/doi/10.1137/07070111X) | SIAM Review | 2009 |

## :blue_book: Miscellaneous
| Title                                                        | Venue      | Year |
| ------------------------------------------------------------ | ---------- | ---- |
| [Very Basics of Tensors with Graphical Notations: Unfolding, Calculations, and Decompositions](https://arxiv.org/abs/2411.16094) | Arxiv | 2024 |
| [Tensor decompositions and their applications](https://tensordec.maths.unitn.it/material/2021_Masterclass_Nick/MCT02-Tucker.pdf) | | 2021 |
| [Nuclear Norm of Higher-Order Tensors](https://arxiv.org/abs/1410.6072) | Mathematics of Computation | 2018 |

## :computer: Repositories
- [TensorKit.jl: A Julia package for large-scale tensor computations, with a hint of category theory](https://arxiv.org/abs/2508.10076)[![GitHub Repo stars](https://img.shields.io/github/stars/Jutho/TensorKit.jl)](https://github.com/Jutho/TensorKit.jl)
- [tednet](https://github.com/tnbar/tednet) [![GitHub Repo stars](https://img.shields.io/github/stars/tnbar/tednet)](https://github.com/tnbar/tednet)
- [TensorLy](https://github.com/tensorly/tensorly) [![GitHub Repo stars](https://img.shields.io/github/stars/tensorly/tensorly)](https://github.com/tensorly/tensorly)
- [TensorLy-Torch](https://tensorly.org/torch/dev/) [![GitHub Repo stars](https://img.shields.io/github/stars/tensorly/torch)](https://github.com/tensorly/torch)
- [PyTorch Tensor Decompositions](https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning)[![GitHub Repo stars](https://img.shields.io/github/stars/jacobgil/pytorch-tensor-decompositions)](https://github.com/jacobgil/pytorch-tensor-decompositions)
- [CNN_compression_with_Tensor_Decomposition](https://github.com/K0EKJE/CNN_compression_with_Tensor_Decomposition)[![GitHub Repo stars](https://img.shields.io/github/stars/K0EKJE/CNN_compression_with_Tensor_Decomposition)](https://github.com/K0EKJE/CNN_compression_with_Tensor_Decomposition)
- [Tensor methods in Python with TensorLy](https://github.com/JeanKossaifi/tensorly-notebooks)[![GitHub Repo stars](https://img.shields.io/github/stars/JeanKossaifi/tensorly-notebooks)](https://github.com/JeanKossaifi/tensorly-notebooks)
- [TensorKrowch: Smooth integration of tensor networks in machine learning](https://arxiv.org/abs/2306.08595)[![GitHub Repo stars](https://img.shields.io/github/stars/joserapa98/tensorkrowch)](https://github.com/joserapa98/tensorkrowch)
- [Python-for-Tensor-Network-Tutorial](https://github.com/ranshiju/Python-for-Tensor-Network-Tutorial)[![GitHub Repo stars](https://img.shields.io/github/stars/ranshiju/Python-for-Tensor-Network-Tutorial)](https://github.com/ranshiju/Python-for-Tensor-Network-Tutorial)
- [Tensor Learning Team](https://qibinzhao.github.io/)
- [Awesome-LoRA](https://github.com/lliai/Awesome-LoRA-Low-Rank-Adaptation) [![GitHub Repo stars](https://img.shields.io/github/stars/lliai/Awesome-LoRA-Low-Rank-Adaptation)](https://github.com/lliai/Awesome-LoRA-Low-Rank-Adaptation)
- [matcouply](https://github.com/MarieRoald/matcouply) [![GitHub Repo stars](https://img.shields.io/github/stars/MarieRoald/matcouply)](https://github.com/MarieRoald/matcouply)
- [torchTT](https://github.com/ion-g-ion/torchTT) [![GitHub Repo stars](https://img.shields.io/github/stars/ion-g-ion/torchTT)](https://github.com/ion-g-ion/torchTT)

## :question: Workshops
- [Tensor Workshops](https://tensorworkshop.github.io/)
- [LoRAINNe’24: workshop on LOw-Rank Approximations and their Interactions with Neural NEtworks](https://cran-simul.github.io/workshop-lorainne-2024/)
- [AAAI'25 workshop on CoLoRAI - Connecting Low-Rank Representations in AI](https://april-tools.github.io/colorai/)
- [Workshop on Low-Rank Models and Applications LRMA 25](https://sites.google.com/view/lrma25)
