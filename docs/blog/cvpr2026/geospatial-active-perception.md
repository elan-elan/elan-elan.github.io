<a class="cvpr-back-link" href="../">Back to CVPR 2026 index</a>

# Geospatial Models And Active Perception

The geospatial lesson from CVPR 2026 should stay calibrated. Foundation models are promising, but the MONTI/CVPR signal was not "big geospatial foundation models solve remote sensing." Smaller models and ImageNet-pretrained baselines can still be competitive in some remote-sensing settings, especially when the benchmark is narrow, labels are policy-shaped, or the output is a simple mask.

The stronger mechanism is active perception: use a global view to decide where high-resolution evidence is worth spending compute, then use the right geospatial output form for the job. Sometimes that output is a polygon, but this batch argues for risk rasters, semantic occupancy, promptable masks, camera poses, and navigation decisions as neighboring structured layers.

<div class="cvpr-flow" aria-label="Geospatial active perception flow">
   <span>global low-resolution scene</span>
   <span>choose evidence or target GSD</span>
   <span>inspect, crop, or register</span>
   <span>emit structured geospatial output</span>
</div>

## Mechanism Map

<div class="mechanism-map" aria-label="Geospatial mechanism map">
   <div><strong>Dataset audit</strong>Detect exact and augmented duplicates before interpreting benchmark scores.</div>
   <div><strong>Resolution-aware encoding</strong>Represent sensors, ground sampling distance, time, and modality explicitly.</div>
   <div><strong>Crop retrieval</strong>Ask for the visual evidence needed for the current query.</div>
   <div><strong>Semantic completion</strong>Predict hidden or future 3D occupancy from aerial views.</div>
   <div><strong>Adaptive training</strong>Revisit hard examples rather than every image every epoch.</div>
   <div><strong>Pixel-to-world alignment</strong>Register camera pixels to a geo-referenced 3D map.</div>
   <div><strong>Risk rasters</strong>Predict spatial risk fields instead of only labels.</div>
</div>

The simplest example is a huge aerial image with one small target. A model should not inspect every pixel at full resolution. It should first find the likely area, zoom in where detail matters, then produce the right output: a crop answer, a mask, a polygon, a pose, or a raster field.

```text
large scene -> coarse search -> high-resolution evidence -> structured map output
```

This is also where aerial imagery becomes a physics problem. A pixel is not just a pixel; it has ground sampling distance, sensor band, acquisition time, atmosphere, lighting, terrain, and view geometry behind it. The best papers on this page make some part of that hidden context explicit.

## Paper-by-paper takeaways

### 🏆 Data Leakage Detection And De-duplication: benchmark hygiene is model progress

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/data-leakage/hash-computation-figure-1.png" alt="Figure 1 from the Data Leakage paper showing the perceptual hash computation pipeline for an aerial image">
   <figcaption>Figure 1 from the Data Leakage paper shows the perceptual-hash pipeline: downsample the image, compute a 32 by 32 discrete cosine transform, keep the low-frequency 8 by 8 block, threshold it by the retained mean, and flatten it into a 64-dimensional hash. This is the core benchmark-hygiene mechanism behind the duplicate checks.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="Data leakage detection summary">
   <div><strong>Problem</strong><span>Large geospatial image datasets can contain exact or augmented duplicates across train, validation, and test splits.</span></div>
   <div><strong>Mechanism</strong><span>Use perceptual hashing to find exact and augmented duplicates without training a new model.</span></div>
   <div><strong>Takeaway</strong><span>AICrowd Mapping Challenge is severely contaminated: 93.45% of validation images and 93.26% of test images are also present in training.</span></div>
   <div><strong>Transfer boundary</strong><span>The method detects image duplication, not label correctness, spatial coverage, or all forms of benchmark leakage.</span></div>
</div>

[Data Leakage Detection and De-duplication in Large Scale Geospatial Image Datasets](https://arxiv.org/abs/2304.02296) is the most important geospatial addition for Aerial2Poly. It is an award-candidate paper that improves evaluation without model training. That matters because polygon-vectorization progress depends on whether the train and test split really measure generalization.

The headline result is blunt. INRIA and SpaceNet 2 have negligible actionable leakage under the paper's checks, but AICrowd Mapping Challenge is badly contaminated. The authors report that 93.45% of validation images and 93.26% of test images are also present in training. After duplicate and augmented-duplicate removal, the train split collapses from 280,741 images to 29,338 unique images before validation leakage removal, then to 15,392 after removing validation leakage.

This is directly tied to Aerial2Poly. AICrowd uses MS-COCO polygonal building annotations, and the paper's first author, Yeshwanth Kumar Adimoolam, is also the lead author and code owner for Pix2Poly. Pix2Poly is one of the closest prior baselines for Aerial2Poly, so this is not a generic data-cleaning aside. It is a warning about the benchmark lineage around polygonal building extraction.

The transfer rule is simple: before claiming a new geospatial vectorizer is better, run leakage and de-dup checks on the splits. A no-training data audit can be more valuable than another model tweak.

### AwaRes: high resolution as a tool call

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/awares/awares-teaser.png" alt="AwaRes workflow using a low-resolution global view and selected high-resolution crops">
   <figcaption>AwaRes first reads a low-resolution overview, then requests only the high-resolution crops needed for the answer.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="AwaRes summary">
   <div><strong>Problem</strong><span>VLMs need high-resolution evidence, but full high-resolution inference is slow and expensive.</span></div>
   <div><strong>Mechanism</strong><span>Low-resolution first turn, structured crop tool calls, SFT for format, then GRPO with answer reward and crop-cost penalties.</span></div>
   <div><strong>Takeaway</strong><span>Resolution should be queried. The model learns when the overview is enough and where detail is worth buying.</span></div>
   <div><strong>Transfer boundary</strong><span>Useful for boundary evidence, but crop policies need tasks where the missing evidence is spatially local.</span></div>
</div>

[AwaRes](https://arxiv.org/abs/2603.16932) is the clearest active-perception template in the batch. It does not merely prune visual tokens after the fact. It trains the model to make a coupled decision: answer from the overview, or call a crop tool for selected high-resolution regions.

The supervision pipeline matters. The paper compares low-resolution and full-resolution answers with an LLM judge, uses an oracle VLM to localize the supporting evidence, maps that evidence to a discrete crop set, trains the tool protocol with SFT, then uses multi-turn GRPO to reduce over-cropping. The report-worthy result is not the exact score, but the behavior shift: after RL, the model becomes much more willing to skip unnecessary crops while preserving full-resolution-level accuracy.

### ★ ZoomEarth: active perception for ultra-high-resolution remote sensing

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/zoomearth/zoomearth-teaser.png" alt="ZoomEarth ultra-high-resolution geospatial active perception examples">
   <figcaption>ZoomEarth turns high-resolution remote-sensing QA into a look-then-crop-then-answer loop. The figure matters because it shows the evidence-acquisition step directly: a model first reasons over the global scene, then spends detail budget only where the answer needs it.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="ZoomEarth summary">
   <div><strong>Problem</strong><span>Ultra-high-resolution scenes are too large for direct VLM reasoning, but downsampling hides small objects and local details.</span></div>
   <div><strong>Mechanism</strong><span>Qwen2.5-VL-3B sees the downsampled scene, predicts a crop region, inspects that crop, then answers.</span></div>
   <div><strong>Takeaway</strong><span>The crop action needs its own reward. Region-Guided reward makes RL learn useful locations despite sparse IoU feedback.</span></div>
   <div><strong>Transfer boundary</strong><span>Best for question-driven inspection. Polygon extraction would need crop rewards tied to boundary or topology quality.</span></div>
</div>

[ZoomEarth](https://arxiv.org/abs/2511.12267) is AwaRes specialized to geospatial scale. The paper builds LRS-GRO from 1,224 ultra-high-resolution images, 3,592 annotated regions, and 13,245 questions across object, region, and global reasoning. SFT teaches the tool format, but the important step is GRPO with a compound reward: answer correctness, region IoU, formatting, and a Region-Guided reward shaped by distance to the target.

The Region-Guided reward is the useful design detail. Pure IoU can be zero for many almost-right crops, so the shaped reward gives a gradient toward the right area. For aerial vectorization, that suggests a similar trick: reward crop selection by distance to unresolved boundary evidence, not just by final polygon score.

### RAMEN: ground sampling distance as a control knob

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/ramen/ramen-mae.png" alt="RAMEN resolution-adjustable multimodal Earth observation encoder architecture">
   <figcaption>RAMEN explicitly models sensor channels, time, and target ground sampling distance before a shared EO transformer.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="RAMEN summary">
   <div><strong>Problem</strong><span>EO data mixes sensors, bands, time intervals, and resolutions; treating every patch like an RGB image throws away physics.</span></div>
   <div><strong>Mechanism</strong><span>Channel-conditioned projection, adjustable spatial resampling by requested GSD, temporal attention, and masked multimodal reconstruction.</span></div>
   <div><strong>Takeaway</strong><span>Resolution is not a nuisance variable. It is an inference-time choice that should match the task scale.</span></div>
   <div><strong>Transfer boundary</strong><span>Useful when metadata is trustworthy. Less helpful when imagery arrives without calibrated GSD, bands, or acquisition time.</span></div>
</div>

[RAMEN](https://arxiv.org/abs/2512.05025) is the best calibration paper in this group. It argues that a geospatial encoder should know which physical channels it sees, when they were acquired, and at what ground sampling distance the downstream feature grid should live. Its adjustable spatial resampler mixes experts using the log ratio between input GSD and target GSD.

The interesting result is not only the PANGAEA average mIoU. It is that the best GSD differs by task: coarse features can help burn scars, while fine features matter for high-resolution segmentation. That is the direct lesson for geospatial foundation models: do not hide resolution decisions inside preprocessing if the task itself lives at a particular physical scale.

### AFSS: train on what is still teaching the detector

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/afss/afss-method.jpg" alt="AFSS adaptive frequency sample selection training scheduler">
   <figcaption>AFSS assigns each image to easy, moderate, or hard sampling regimes using a per-image learning-sufficiency score. The schedule is important because it revisits easy images often enough to avoid forgetting while keeping hard examples in the active loop.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="AFSS summary">
   <div><strong>Problem</strong><span>Detector training wastes epochs on examples the model already handles, especially in large aerial datasets.</span></div>
   <div><strong>Mechanism</strong><span>Per-image sufficiency is `min(precision, recall)`, then easy images are reviewed rarely, moderate images partially, and hard images always.</span></div>
   <div><strong>Takeaway</strong><span>Adaptive sampling is anti-forgetting, not deletion. Easy images still return on a forced review schedule.</span></div>
   <div><strong>Transfer boundary</strong><span>The schedule depends on stable per-image metrics. Polygon tasks need comparable precision/recall or topology proxies.</span></div>
</div>

[AFSS](https://arxiv.org/abs/2603.17684) is not a geospatial foundation model, but it is operationally useful. It asks whether YOLO really needs to see every image in every epoch. The answer is no, if the sampler tracks whether each image is still teaching the model.

The key formula is deliberately simple: learning sufficiency is `min(precision, recall)`. Images above the easy threshold get sparse review, moderate images get periodic coverage, and hard images stay in the active set. For DOTA and DIOR-R oriented bounding boxes, the paper reports more than 1.63x speedups while maintaining or improving accuracy.

### ★ SegEarth-R2: language-guided masks need dynamic mask queries

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/segearth-r2/segearth-r2-framework.png" alt="SegEarth-R2 language-guided remote-sensing segmentation framework">
   <figcaption>SegEarth-R2 turns [SEG] tokens into dynamic mask queries and supervises their spatial attention. The framework shows why language-guided remote-sensing segmentation should expose the requested objects explicitly instead of hiding them inside a fixed proposal bank.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="SegEarth-R2 summary">
   <div><strong>Problem</strong><span>Remote-sensing segmentation prompts vary by object granularity, number of targets, reasoning need, and wording.</span></div>
   <div><strong>Mechanism</strong><span>MLLM [SEG] tokens become dynamic Mask2Former queries; spatial attention supervision aligns [SEG] attention with target masks.</span></div>
   <div><strong>Takeaway</strong><span>The model should produce as many mask queries as the instruction requires, not always propose 100 masks and select later.</span></div>
   <div><strong>Transfer boundary</strong><span>Strong for language-guided masks, but polygon vectorization still needs ordering, simplification, and topology constraints.</span></div>
</div>

[SegEarth-R2](https://arxiv.org/abs/2512.20013) is the most directly relevant segmentation paper in the batch. It contributes LaSeRS and a model that binds language to masks more explicitly than generic promptable segmentation. The subtle mechanism is spatial attention supervision: [SEG]-token attention maps are trained against the target masks, which helps small or component-level objects.

The dynamic-query result is also worth carrying forward. In one ablation, using one dynamic query instead of the default 100 improves gIoU while reducing inference time and TFLOPs. That is a useful warning for vectorized geospatial output: dense heads should expose the number and identity of requested outputs, not bury them in a fixed proposal set.

### ★ PiLoT: every pixel can become a geographic ray

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/pilot/pilot-overview.png" alt="PiLoT UAV pixel-to-3D registration and target geolocalization overview">
   <figcaption>PiLoT registers live UAV video to a geo-referenced 3D map, enabling both ego-localization and target geolocation. The figure matters because it makes pixel-to-world alignment a model output: once pose is known, a target pixel can become a geographic ray.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="PiLoT summary">
   <div><strong>Problem</strong><span>GNSS/IMU can fail, drift, or be unavailable, but UAV imagery still needs geographic pose and target coordinates.</span></div>
   <div><strong>Mechanism</strong><span>A dual-thread renderer/localizer matches live monocular frames to a geo-referenced 3D map with neural features and pose optimization.</span></div>
   <div><strong>Takeaway</strong><span>Pixel-to-world registration is a first-class geospatial primitive, not only a robotics afterthought.</span></div>
   <div><strong>Transfer boundary</strong><span>It assumes a usable 3D map and a coarse first-frame prior; smaller-scale deployments may not always have that map.</span></div>
</div>

[PiLoT](https://arxiv.org/abs/2603.20778) sits in the pixel-to-world alignment lane. Instead of extracting shapes from one image, it registers a live UAV stream against a geo-referenced 3D map. A rendering thread provides reference views; a localization thread uses a MobileOne-UNet feature pyramid and a Joint Neural-Guided Stochastic-Gradient Optimizer with rotation-aware pose hypotheses and coarse-to-fine LM refinement.

The bridge to geospatial decision systems is the output interface. Once camera pose is known, a target pixel can be ray-cast into geographic space. That makes it natural to connect visual detections, footprints, hazards, and navigation decisions to the same map frame.

### ★ HTNav: split aerial navigation into route and action

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/htnav/htnav-architecture.png" alt="HTNav hybrid tiered aerial vision-and-language navigation architecture">
   <figcaption>HTNav separates macro waypoint planning from low-level aerial actions, then fine-tunes with PPO. The architecture shows a useful output split for aerial reasoning: choose the route-level subgoal first, then choose the local action.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="HTNav summary">
   <div><strong>Problem</strong><span>Aerial VLN requires long-range route reasoning and local action control under language instructions.</span></div>
   <div><strong>Mechanism</strong><span>Imitation learning initializes a value-aware policy; PPO then fine-tunes a MacroPlanner plus MicroActor controller.</span></div>
   <div><strong>Takeaway</strong><span>Long-horizon aerial reasoning benefits from a tiered output structure: waypoint first, action second.</span></div>
   <div><strong>Transfer boundary</strong><span>Navigation metrics remain far from human performance, and dataset landmark quality can dominate apparent progress.</span></div>
</div>

[HTNav](https://arxiv.org/abs/2604.08883) is the navigation counterpart to the crop-and-register papers. It first trains by imitation, then uses PPO for reinforcement learning. The architecture is intentionally tiered: a MacroPlanner predicts sub-goals from map features, pose, and language, while a MicroActor turns visual observation and the current sub-goal into one of six discrete actions.

The paper also audits CityNav, correcting roughly 800 landmark annotations and removing 311 non-navigable cases. That matters for report tone. Progress on aerial reasoning is partly model architecture and partly dataset hygiene.

### OccuFly: a benchmark for hidden aerial 3D structure

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/occufly/occufly-framework.png" alt="OccuFly LiDAR-free aerial semantic scene completion benchmark pipeline">
   <figcaption>OccuFly builds aerial semantic scene completion labels from RGB flight videos using SfM, MVS, sparse labels, and class-aware densification. The figure matters because it shows a LiDAR-free path from ordinary flight video to hidden 3D occupancy supervision.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="OccuFly summary">
   <div><strong>Problem</strong><span>Remote-sensing benchmarks often evaluate visible objects and masks, not hidden semantic 3D structure.</span></div>
   <div><strong>Mechanism</strong><span>LiDAR-free reconstruction plus sparse 2D annotation, 2D-to-3D label lifting, label propagation, and class-aware voxel densification.</span></div>
   <div><strong>Takeaway</strong><span>The benchmark exposes a gap: aerial semantic scene completion remains weak even when occupancy IoU is nontrivial.</span></div>
   <div><strong>Transfer boundary</strong><span>It is adjacent to polygon extraction, not a replacement. It measures implied 3D scene structure, not visible building footprints.</span></div>
</div>

[OccuFly](https://arxiv.org/abs/2512.20770) matters because it gives the geospatial section a benchmark gap. Standard aerial extraction usually focuses on visible masks, boxes, or vectors. OccuFly asks what semantic 3D occupancy the model can complete from an aerial view, including structure that is occluded or contextually implied.

The construction is practical: reconstruct metric point clouds and depths with SfM/MVS, manually label under 10% of images, lift labels into 3D through correspondences, propagate by majority and kNN voting, then densify class-aware voxels. The current baseline numbers are low enough to be useful. They say this capability is not solved.

### FireScope: reasoning is wired into the risk raster

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/firescope/firescope-method.png" alt="FireScope GRPO-trained chain-of-thought oracle and FiLM-conditioned wildfire risk raster decoder">
   <figcaption>FireScope fine-tunes a CoT Oracle with GRPO, then uses the scalar risk estimate to FiLM-condition a raster decoder.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="FireScope summary">
   <div><strong>Problem</strong><span>Wildfire risk is spatial and continuous, so a class label is too thin for operational geospatial risk modeling.</span></div>
   <div><strong>Mechanism</strong><span>Qwen2.5-VL Oracle produces chain-of-thought reasoning plus scalar ordinal risk; the scalar FiLM-conditions a 341x341 raster decoder.</span></div>
   <div><strong>Takeaway</strong><span>The CoT Oracle is not decoration. GRPO trains the reasoning/risk bottleneck that conditions the final raster field.</span></div>
   <div><strong>Transfer boundary</strong><span>CoT text should be audited carefully; the durable transfer is the structured risk-raster output and risk-conditioned decoder interface.</span></div>
</div>

[FireScope](https://arxiv.org/abs/2511.17171) is directly relevant to spatial risk mapping. FireScope-Bench pairs Sentinel-2 imagery and NASA POWER climate vectors with expert wildfire-risk rasters, then adds European event/control examples for out-of-distribution evaluation. The model has two stages: a Qwen2.5-VL-7B Oracle and a lightweight raster generator.

The important correction is the GRPO step. The Oracle is fine-tuned with GRPO to produce explicit chain-of-thought reasoning and a scalar ordinal risk estimate. That scalar does real work: it conditions the raster encoder-decoder through FiLM, and the decoder predicts a continuous 341x341 risk raster using Smooth-L1, SSIM, and gradient-edge losses. The transferable pattern is to use imagery and context to produce a spatial risk field, not merely a category.

## The Reality Check

The report should not overclaim geospatial foundation models. Remote-sensing imagery is not just natural imagery from above. Sensors differ, ground sampling distance changes the meaning of a pixel, seasonal and atmospheric conditions matter, and labels often reflect geography or policy as much as visual appearance.

That makes a practical rule:

<div class="cvpr-technique-callout">
   <strong>Do not assume scale wins automatically.</strong>
   <ol>
      <li>Run split leakage and de-dup checks before trusting benchmark gains.</li>
      <li>Compare against small task-specific models and ImageNet-pretrained baselines.</li>
      <li>Separate representation gains from resolution, tiling, postprocessing, and label-policy gains.</li>
      <li>Evaluate on the operational output: boundaries, risk rasters, semantic completion, or decisions.</li>
   </ol>
</div>

## Sources

Public anchors: [Data Leakage Detection and De-duplication in Large Scale Geospatial Image Datasets](https://arxiv.org/abs/2304.02296), [AwaRes](https://arxiv.org/abs/2603.16932), [ZoomEarth](https://arxiv.org/abs/2511.12267), [RAMEN](https://arxiv.org/abs/2512.05025), [AFSS](https://arxiv.org/abs/2603.17684), [SegEarth-R2](https://arxiv.org/abs/2512.20013), [PiLoT](https://arxiv.org/abs/2603.20778), [HTNav](https://arxiv.org/abs/2604.08883), [OccuFly](https://arxiv.org/abs/2512.20770), and [FireScope](https://arxiv.org/abs/2511.17171).