<a class="cvpr-back-link" href="../">Back to CVPR 2026 index</a>

# 3D Geometry And Reconstruction

<p class="blog-post-date">2026-06-08</p>

The most useful 3D papers at CVPR 2026 were not just better reconstruction demos. They treated geometry as reusable memory. A model can look at images or video, build a state that stores depth, camera pose, material, or motion, and let later tasks query that state.

That framing matters for geospatial AI and visual inspection because many decisions depend on scale, layout, material, pose, lighting, and physical consistency. A caption can say "window" or "roof." A useful vision system has to understand whether the glass is reflective, whether the camera pose is plausible, whether a measurement is metric, and whether a predicted structure stays stable across views.

<div class="cvpr-flow" aria-label="3D reconstruction as reusable scene state">
   <span>image or video</span>
   <span>geometry-aware memory</span>
   <span>metric or material state</span>
   <span>task-specific readout</span>
</div>

## Mechanism Map

<div class="mechanism-map" aria-label="3D mechanism map">
   <div><strong>Feed-forward scene memory</strong>Predict cameras, depth, and reusable spatial tokens directly from images or video.</div>
   <div><strong>Candidate data engines</strong>Generate 3D proposals, then ask humans to verify, rank, or repair instead of starting from scratch.</div>
   <div><strong>Queryable 4D state</strong>Turn dynamic reconstruction into a memory that answers targeted spatiotemporal queries.</div>
   <div><strong>Metric backends</strong>Add scale and global refinement behind fast foundation reconstruction models.</div>
   <div><strong>Hard-material modeling</strong>Test and model glass, reflections, transparency, low texture, and relighting explicitly.</div>
   <div><strong>Structured 3D latents</strong>Represent geometry and material attributes natively rather than flattening 3D into image views.</div>
</div>

The shared pattern is simple:

```text
visual capture -> geometry memory -> decision-facing structure
```

The mistake would be to judge these papers only by prettier meshes. The stronger question is whether their intermediate geometry can make downstream decisions more stable, measurable, and inspectable.

Read this page with a concrete image in mind: two photos of the same roof corner from different angles. A weak model says "roof." A geometry model should also estimate where the camera moved, which corner is the same physical point, how far away it is, and whether a shiny patch is surface material or reflected light.

For each paper, the simple checklist is:

- What is stored: depth, camera pose, point tracks, Gaussians, mesh tokens, or material state?
- What changes it: attention, a backend optimizer, a data engine, or a renderer?
- What keeps it valid: metric scale, multi-view consistency, physics, or human verification?
- What can a later task ask from it?

## Paper-by-paper takeaways

### 🏆 VGGT-Omega: scale through register attention

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/vggt-omega/vggt-omega-architecture.png" alt="VGGT-Omega architecture with alternating global and register attention for feed-forward reconstruction">
   <figcaption>VGGT-Omega routes global 3D context through compact scene/register tokens. The figure matters because it shows the section's central idea: geometry can be stored as reusable memory before a downstream head asks for cameras, depth, or task-specific evidence.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="VGGT-Omega summary">
   <div><strong>Problem</strong><span>Feed-forward reconstruction wants more frames, more data, and dynamic video, but full global attention and dense heads become too expensive.</span></div>
   <div><strong>Mechanism</strong><span>DINOv3 backbone, one camera token and 16 scene/register tokens per frame, 25% register-attention replacement, lighter dense head, and teacher-student video self-supervision.</span></div>
   <div><strong>Takeaway</strong><span>The useful product is not just depth. It is a geometry-aware memory that can be reused by other task heads.</span></div>
   <div><strong>Transfer boundary</strong><span>Nadir aerial imagery may not contain enough 3D parallax by itself; the same token idea may need to learn layout, height priors, or multi-view consistency.</span></div>
</div>

[VGGT-Omega](https://arxiv.org/abs/2605.15195) is the backbone anchor for this page. It changes the attention pattern so scaling is practical: some expensive frame-wide attention is replaced by register attention through compact scene tokens, while a lighter dense prediction head keeps the reconstruction losses but reduces memory pressure.

For downstream spatial reasoning, the same memory could carry layout, scale, or structural context before a task head predicts the final evidence.

### 🏆 SAM 3D: data engines are part of the model

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/sam-3d/sam-3d-teaser.jpg" alt="SAM 3D examples of visually grounded 3D object reconstruction from natural images">
   <figcaption>SAM 3D reconstructs geometry, texture, and layout from ordinary masked image inputs. The teaser shows the public-facing capability, but the more transferable lesson is the data engine behind it.</figcaption>
</figure>

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/sam-3d/sam-3d-data-engine.png" alt="SAM 3D data engine with generation model, human verification, ranking, alignment, and specialist repair loops">
   <figcaption>SAM 3D's data engine turns hard 3D annotation into a loop: the model proposes candidate geometry and texture, humans verify or rank outputs, accepted examples improve alignment, and hard cases route to specialist repair. This is why the paper is not only a reconstruction model; it is also an annotation system for scaling visually grounded 3D supervision.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="SAM 3D summary">
   <div><strong>Problem</strong><span>Real-world 3D labels are expensive, especially for natural images with occlusion, clutter, and long-tail object categories.</span></div>
   <div><strong>Mechanism</strong><span>Rectified conditional flow matching over 3D modalities plus a candidate-generation, human verification, preference, and specialist repair loop.</span></div>
   <div><strong>Takeaway</strong><span>The data engine is the transferable idea: make hard structured labels cheaper by ranking and verifying model candidates.</span></div>
   <div><strong>Transfer boundary</strong><span>SAM 3D assumes a target mask and object-centric reconstruction; full scenes need multi-object context and failure propagation checks.</span></div>
</div>

[SAM 3D](https://arxiv.org/abs/2511.16624) is easy to summarize as "3Dfy anything," but the more useful lesson is operational. The system generates candidate shapes and textures, asks humans to verify or rank them, uses accepted examples and preferences for alignment, and sends hard cases to specialist artists. That is how it turns an expensive annotation problem into a repeated improvement loop.

For visual inspection, it suggests the same pattern for hard labels such as material state, visible defects, or room layout.

### 🏆 D4RT: reconstruction as a query interface

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/d4rt/d4rt-architecture.png" alt="D4RT architecture with global scene representation and lightweight query decoder">
   <figcaption>D4RT separates the global 4D scene representation from a decoder that answers targeted spatial and temporal queries. The figure is useful because it makes reconstruction look less like a mesh export and more like a queryable memory.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="D4RT summary">
   <div><strong>Problem</strong><span>Dynamic-scene reconstruction often wastes compute predicting every possible point when downstream tasks need only selected facts.</span></div>
   <div><strong>Mechanism</strong><span>A global scene representation plus independent query decoder; queries specify source pixel, target time, and camera time.</span></div>
   <div><strong>Takeaway</strong><span>Depth, tracks, cameras, and correspondences become different questions asked of the same 4D memory.</span></div>
   <div><strong>Transfer boundary</strong><span>The abstraction is strongest when the downstream question is localized; dense global outputs still require careful query scheduling.</span></div>
</div>

[D4RT](https://arxiv.org/abs/2512.08924) reframes dynamic reconstruction as a service: build the scene memory once, then ask for the spatiotemporal fact you need. A query can request where a pixel lands at another time, what depth a target point has, or how camera and scene motion relate.

That is a good video-understanding mental model. A handheld walkthrough does not always need a perfect full mesh. It may need targeted facts: wall plane consistency, window position over time, camera trajectory plausibility, or whether a suspected visual cue stays fixed to a surface.

### ★ AMB3R: metric scale needs a backend

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/amb3r/amb3r-backend.png" alt="AMB3R backend projecting normalized point maps into a sparse voxel transformer for metric refinement">
   <figcaption>AMB3R keeps a frozen feed-forward front end and adds a sparse voxel backend for metric-scale refinement. The backend is the bridge from visually plausible reconstruction to measurement-facing geometry.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="AMB3R summary">
   <div><strong>Problem</strong><span>Foundation reconstruction can look plausible while still missing metric scale and global geometric consistency.</span></div>
   <div><strong>Mechanism</strong><span>Frozen VGGT-style front end, metric scale head, sparse voxel projection, Point Transformer v3 U-Net refinement, and zero-conv fusion back into the decoder.</span></div>
   <div><strong>Takeaway</strong><span>The backend is a practical bridge from reconstruction as visual memory to reconstruction as measurement.</span></div>
   <div><strong>Transfer boundary</strong><span>The metric promise depends on calibration, data distribution, and scale supervision quality; it should be evaluated by downstream measurement error.</span></div>
</div>

[AMB3R](https://arxiv.org/abs/2511.20343) is the "make it measurable" paper. It does not throw away the feed-forward reconstruction backbone. It adds a metric scale head and a sparse voxel backend that aggregates point-map features, refines them with a 3D transformer, and feeds the refined representation back into image-space decoding.

For measurement-oriented vision, that distinction matters. A beautiful reconstruction is not enough if the model cannot support measurements, clearances, scale, or relative positions. AMB3R is a reminder to judge geometry by what can be measured from it.

### 🏆 3DReflecNet: the failure set is glass, metal, and low texture

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/3dreflecnet/3dreflecnet-pipeline.png" alt="3DReflecNet dataset construction and evaluation pipeline for reflective transparent and low-texture objects">
   <figcaption>3DReflecNet builds a hybrid synthetic/real benchmark around surfaces that violate ordinary multi-view reconstruction assumptions: glass, reflective materials, transparency, and low texture. The figure matters because these are ordinary inspection surfaces, not exotic edge cases.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="3DReflecNet summary">
   <div><strong>Problem</strong><span>SfM and MVS assume stable appearance, straight light paths, and enough texture; reflective, transparent, and low-texture surfaces break those assumptions.</span></div>
   <div><strong>Mechanism</strong><span>Hybrid dataset with over 12,000 synthetic objects, more than 1,000 real scans, PBR material variation, lighting sweeps, and multi-task evaluation.</span></div>
   <div><strong>Takeaway</strong><span>A reconstruction benchmark should include the exact materials where inspection systems fail quietly.</span></div>
   <div><strong>Transfer boundary</strong><span>It is object-centric and benchmark-oriented; field deployment still needs scene context, weather, and capture-protocol variation.</span></div>
</div>

[3DReflecNet](https://arxiv.org/abs/2605.10204) is the reality check. It targets reflective, transparent, and low-texture objects because those cases break photometric consistency and feature matching. The dataset combines physically based synthetic rendering, generated 3D assets, and real captures with annotations for reconstruction and related tasks.

For visual inspection, this is not an edge case. Windows, polished metal, mirrors, wet surfaces, blank walls, glossy surfaces, and low-texture materials are ordinary. A robust evaluation set should include these conditions before anyone trusts reconstruction-derived decisions.

### ★ TokenGS: learnable geometry tokens instead of pixel-bound Gaussians

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/tokengs/tokengs-architecture.png" alt="TokenGS encoder decoder with learnable Gaussian tokens decoupled from input pixels">
   <figcaption>TokenGS lets learnable Gaussian tokens cross-attend to image features, decoupling primitive count from image resolution and view count. The diagram shows why tokenized structure can be predicted as its own scene representation rather than tied one-to-one to pixels.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="TokenGS summary">
   <div><strong>Problem</strong><span>Pixel-aligned Gaussian prediction binds 3D primitive count and position to image resolution, camera rays, and observed pixels.</span></div>
   <div><strong>Mechanism</strong><span>Posed image encoder with Plucker coordinates, learnable 3DGS tokens, direct canonical XYZ regression, visibility loss, and test-time token tuning.</span></div>
   <div><strong>Takeaway</strong><span>Geometry tokens can be a structured intermediate representation rather than a byproduct of pixels.</span></div>
   <div><strong>Transfer boundary</strong><span>The model still needs posed inputs and rendering losses; tokenized structure alone does not solve ordering or topology for polygon outputs.</span></div>
</div>

[TokenGS](https://arxiv.org/abs/2604.15239) is especially relevant to structured decoding. The model predicts a set of Gaussian tokens that are not one-to-one with pixels. Those tokens cross-attend to compact image features and directly regress 3D Gaussian means in a canonical coordinate space. A visibility loss reduces ghost primitives in unobserved regions, and test-time token tuning improves the scene without discarding the learned prior.

The analog is not "use Gaussians for every structured task." It is the decoupling move: let learned structure tokens gather scene evidence before a later module produces a task-specific output.

### ★ RT-Splatting: reflections and transmission need separate knobs

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/rt-splatting/rt-splatting-overview.png" alt="RT-Splatting overview with occupancy opacity factorization and hybrid surface volume rendering">
   <figcaption>RT-Splatting factorizes geometric occupancy from optical opacity so one Gaussian set can model both reflective surfaces and transmission. The overview matters because glass-like materials should not be forced into a single opacity meaning when reflection and transmitted background content coexist.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="RT-Splatting summary">
   <div><strong>Problem</strong><span>Semi-transparent specular surfaces mix reflection and transmission, so ordinary 3DGS tends to blur reflections or over-occlude background content.</span></div>
   <div><strong>Mechanism</strong><span>Occupancy-opacity factorization, deferred reflective surface pass, forward transmission pass, dynamic attenuation, and specular-aware gradient gating.</span></div>
   <div><strong>Takeaway</strong><span>Glass-like surfaces should not be forced into a single opacity meaning.</span></div>
   <div><strong>Transfer boundary</strong><span>The paper focuses on thin semi-transparent surfaces and does not fully model thick refraction or multiple light bounces.</span></div>
</div>

[RT-Splatting](https://arxiv.org/abs/2605.18263) is a clean mechanism paper for a common visual problem: windows and glass surfaces are both visible and see-through. The method lets a Gaussian have high geometric occupancy but low optical opacity, so it can contribute to a reflective surface representation without simply blocking the ray.

That is useful beyond rendering. A vision model should know when a visual cue belongs to the surface, when it is a reflection, and when it is transmitted content behind glass. Otherwise it can turn lighting artifacts into false evidence.

### ★ GeoRelight: geometry and lighting should condition each other

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/georelight/georelight-pipeline.png" alt="GeoRelight flexible multi-modal diffusion transformer for relighting and reconstruction">
   <figcaption>GeoRelight repurposes a video diffusion transformer's temporal axis as a modality axis for relighting, normals, albedo, segmentation, and geometry. The figure shows the key trick: the same network can treat different physical scene factors as conditions or generated targets.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="GeoRelight summary">
   <div><strong>Problem</strong><span>Single-image relighting and geometry estimation are coupled: shape affects shadows, while shading provides shape cues.</span></div>
   <div><strong>Mechanism</strong><span>Flexible multi-modal DiT, modality switch mask, global image and illumination conditions, and isotropic normalized orthographic depth for geometry.</span></div>
   <div><strong>Takeaway</strong><span>Appearance and geometry should be solved together when the decision depends on what is material versus what is illumination.</span></div>
   <div><strong>Transfer boundary</strong><span>The current paper is single-image and human-centric; broader scenes need temporal consistency, broader object coverage, and calibrated capture conditions.</span></div>
</div>

[GeoRelight](https://arxiv.org/abs/2604.20715) is not a remote-sensing model, but it has a strong mechanism transfer. It uses a video DiT as a flexible multimodal generator: relit image, albedo, normal, segmentation, and geometry can be targets or conditions. A switch mask tells the model which modalities to generate.

For inspection, the value is disentanglement. If a wall looks dark, is that shadow, material color, moisture, or damage? A model that jointly reasons about geometry, lighting, and appearance has a better chance of separating those causes than a classifier trained only on RGB.

### 🏆 Native Structured Latents: 3D generation needs native structure

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/structured-latents/structured-latents-overview.png" alt="Native structured 3D latent pipeline with sparse compression VAE and flow matching generation">
   <figcaption>Native Structured Latents compress 3D geometry and PBR appearance into compact sparse latents before large-scale 3D generation. The overview is important because it keeps topology, occupancy, and material attributes native instead of flattening them through rendered views.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="Native Structured Latents summary">
   <div><strong>Problem</strong><span>Many 3D generation systems flatten 3D through views or weak proxies, losing topology, material, and high-resolution structure.</span></div>
   <div><strong>Mechanism</strong><span>O-Voxel representation, flexible dual grid, active voxels with geometry and PBR attributes, sparse compression VAE, and flow-matching generation.</span></div>
   <div><strong>Takeaway</strong><span>If the target is 3D, the latent should preserve geometry and materials natively.</span></div>
   <div><strong>Transfer boundary</strong><span>This is a generation representation, not an inspection model; downstream tasks still need grounding, uncertainty, and capture evidence.</span></div>
</div>

[Native and Compact Structured Latents for 3D Generation](https://arxiv.org/abs/2512.14692) adds a different axis to the section. Its O-Voxel representation stores surface geometry and physically based appearance attributes in a sparse structured grid, then compresses that structure with a VAE so large flow-matching models can generate high-resolution assets.

The relevant principle is representational honesty. If the output has topology, material, or geometry constraints, the latent should not hide them inside a flat image token stream. For structured visual reasoning, it points toward compact 3D state that can carry geometry and material attributes together.

### ★ MoRe: motion-aware reconstruction for streaming video

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/more/more-pipeline.png" alt="MoRe motion-aware feed-forward 4D reconstruction transformer pipeline">
   <figcaption>MoRe uses motion-aware attention supervision during training, then reconstructs dynamic 4D scenes from streaming video without masks at inference. The mechanism is practical for walkthroughs because it teaches camera tokens to prefer static evidence while moving objects remain present.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="MoRe summary">
   <div><strong>Problem</strong><span>Camera estimation in dynamic video is confused when moving objects dominate attention and static background evidence is diluted.</span></div>
   <div><strong>Mechanism</strong><span>Training-time motion-mask attention forcing, grouped causal attention for streaming, cached token aggregation, and BA-like camera refinement.</span></div>
   <div><strong>Takeaway</strong><span>Motion should supervise where camera tokens look, not just be predicted after the fact.</span></div>
   <div><strong>Transfer boundary</strong><span>The training signal depends on motion-mask quality, and long-term temporal dependencies remain difficult.</span></div>
</div>

[MoRe](https://arxiv.org/abs/2603.05078) deserves to be promoted into this page because phone walkthroughs are rarely static. During training, the model uses motion masks to push camera-token attention toward static regions, but it does not require masks at inference. Grouped causal attention keeps streaming feasible, then a refinement stage aggregates cached information for camera updates.

For handheld capture, that is a practical design. People move, doors swing, reflections shift, and the camera itself is unstable. A reconstruction model that explicitly learns to separate motion from camera evidence is closer to a usable visual-understanding pipeline than a static-only model.

## VGGT-Omega Versus SAM 3D

These two papers remain the section's core contrast.

| Question | VGGT-Omega answer | SAM 3D answer |
| --- | --- | --- |
| What is the reusable asset? | Geometry-aware scene/register tokens and a reconstruction backbone. | A model-in-the-loop data engine for visually grounded 3D labels. |
| Where does scale come from? | More data, register attention, lighter dense heads, and self-supervised video training. | Candidate generation, human verification, preference data, and specialist repair of hard cases. |
| What is reusable? | Geometry tokens for layout, scale, pose, and structural context. | Efficient annotation of hard visual or 3D evidence. |

VGGT-Omega says geometry can be a backbone. SAM 3D says data collection can be a flywheel. The report needs both ideas: reusable spatial representation and a practical way to improve it when direct annotation is too expensive.

## Sources

Public paper links: [VGGT-Omega](https://arxiv.org/abs/2605.15195), [SAM 3D](https://arxiv.org/abs/2511.16624), [D4RT](https://arxiv.org/abs/2512.08924), [AMB3R](https://arxiv.org/abs/2511.20343), [3DReflecNet](https://arxiv.org/abs/2605.10204), [TokenGS](https://arxiv.org/abs/2604.15239), [RT-Splatting](https://arxiv.org/abs/2605.18263), [GeoRelight](https://arxiv.org/abs/2604.20715), [Native Structured Latents](https://arxiv.org/abs/2512.14692), and [MoRe](https://arxiv.org/abs/2603.05078).