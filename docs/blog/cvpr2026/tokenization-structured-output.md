<a class="cvpr-back-link" href="../">Back to CVPR 2026 index</a>

# Tokenization And Structured Output

<p class="blog-post-date">2026-06-08</p>

This is the most Aerial2Poly-native question in the report: what alphabet should geometry speak?

Pix2Seq, Pix2Seq v2, Pix2Poly, and the local Aerial2Poly framework all share the same core move: turn visual structure into a parseable token language. The lineage matters, but it does not need to dominate this page. CVPR 2026 pushes the next question: once geometry already speaks in tokens, what alphabet, ordering policy, token budget, and topology backend make the language reliable?

<div class="cvpr-flow" aria-label="Structured-output token flow">
   <span>image evidence</span>
   <span>visual alphabet</span>
   <span>decoding policy</span>
   <span>geometry tokens</span>
   <span>valid vector output</span>
</div>

## Mechanism Map

<div class="mechanism-map" aria-label="Structured output mechanism map">
   <div><strong>Geometry as language</strong>Pix2Seq, Pix2Seq v2, Pix2Poly, and Aerial2Poly make coordinate, class, separator, and stop tokens a visual output contract.</div>
   <div><strong>Alphabet design</strong>AToken, VectorArk, and LottieGPT show that the token vocabulary is a modeling decision, not a file-format afterthought.</div>
   <div><strong>Order policy</strong>FVAR reframes visual AR as next-focus prediction; the same idea can become next building, ring, or boundary segment.</div>
   <div><strong>Token budget</strong>VibeToken makes latent length a quality/compute knob instead of letting resolution dictate sequence cost.</div>
   <div><strong>Topology backend</strong>ACPV-Net turns semantic and vertex evidence into a planar graph, enforcing shared-edge consistency after prediction.</div>
   <div><strong>Scene geometry tokens</strong>PixARMesh extends the sequence idea to object poses and mesh-native 3D scene geometry.</div>
</div>

The common pattern is:

```text
visual evidence + representation grammar + decoding schedule -> parseable structure
```

This means the tokenizer and output grammar are not clerical choices. They decide which errors the model can express, which errors a verifier can catch, and which errors silently become invalid polygons.

Think of vector output as a drawing written as a sentence. The sentence has to be readable by both the model and a geometry parser.

```text
CLASS_BUILDING X_012 Y_044 X_019 Y_046 X_020 Y_052 SEP EOS
```

The page asks three plain questions: what is the alphabet, what order should the sentence be written in, and what rule catches invalid geometry before it becomes a map error?

## Lineage In One Paragraph

[Pix2Seq](https://arxiv.org/abs/2109.10852) made detection look like translation by serializing boxes and classes into tokens. [Pix2Seq v2](https://arxiv.org/abs/2206.07669) showed the same sequence interface could cover multiple visual tasks with prompt-selected output contracts. [Pix2Poly](https://arxiv.org/abs/2412.07899) moved that idea closer to building footprints by predicting vertices and recovering connectivity. Aerial2Poly follows the same geometry-as-language line, but this page now focuses on the CVPR 2026 techniques that stretch the alphabet, ordering, budget, and topology machinery beyond that lineage.

## New Alphabets

### AToken: a shared visual alphabet across images, videos, and 3D

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/atoken/atoken-overview.png" alt="AToken overview showing a unified tokenizer with sparse 4D visual latents for images, videos, and 3D assets">
   <figcaption>AToken learns sparse 4D latent tokens that can serve reconstruction and understanding across images, videos, and 3D assets. The overview matters because it treats tokenizer design as the core model interface, not a preprocessing detail.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="AToken summary">
   <div><strong>Problem</strong><span>Visual tokenizers often specialize by modality or optimize reconstruction while weakening semantic utility.</span></div>
   <div><strong>Mechanism</strong><span>Use sparse 4D latents, 4D RoPE, reconstruction and understanding projections, Gram/perceptual losses, and a progressive image-video-3D curriculum.</span></div>
   <div><strong>Takeaway</strong><span>A useful visual alphabet can be shared if position, modality, reconstruction, and semantics are all represented explicitly.</span></div>
   <div><strong>Transfer boundary</strong><span>The paper is not a vector-map method; it is an interface-design paper for visual evidence.</span></div>
</div>

[AToken](https://arxiv.org/abs/2509.14476) is useful because it treats tokenization as the main product. It asks what kind of latent can carry images, videos, and 3D assets while still supporting both reconstruction and recognition.

The transfer is not to rebuild AToken. It is to stop thinking of the image encoder, class prompts, coordinate embeddings, and geometry tokens as unrelated objects. A shared interface could let the model reason about image evidence and structured output inside one consistent token space.

### VectorArk: vector graphics through rounded-polygon grammar

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/vectorark/vectorark-training-inference.png" alt="VectorArk training and inference diagram showing rounded polygon representation for image vectorization">
   <figcaption>VectorArk replaces raw SVG path commands with rounded-polygon triples, then trains and ranks candidates in that constrained vector language. The figure shows the grammar decision that makes outputs shorter, more editable, and easier to score than arbitrary SVG command streams.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="VectorArk summary">
   <div><strong>Problem</strong><span>Raw SVG commands are expressive, but unstable and verbose for learned image vectorization.</span></div>
   <div><strong>Mechanism</strong><span>Represent paths as rounded polygon triples `(x, y, d)`, normalize with outline rasters, add degraded-input training, and select among sampled candidates.</span></div>
   <div><strong>Takeaway</strong><span>The right primitive set can make generation shorter, more editable, and more geometrically biased.</span></div>
   <div><strong>Transfer boundary</strong><span>Graphic icons and aerial buildings have different priors; rounded corners should be tested as an option, not assumed.</span></div>
</div>

[VectorArk](https://arxiv.org/abs/2605.24398) is the cleanest alphabet-design neighbor. It does not ask a model to emit arbitrary SVG command soup. It compresses curves and corners into a practical rounded-polygon representation, then trains the model inside that grammar.

That turns into a representation question: compare plain coordinate tokens against richer polygon primitives. Maybe most buildings only need straight edges and right angles. Maybe some neighborhoods need rounded-corner or arc-like tokens. The only honest answer is an ablation that measures vertex economy, topology errors, and editability together.

### LottieGPT: animation as a structured token language

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/lottiegpt/lottiegpt-pipeline.png" alt="LottieGPT pipeline showing tokenization of vector animation structure for autoregressive generation">
   <figcaption>LottieGPT tokenizes vector animation hierarchy, properties, keyframes, and easing curves instead of generating frame pixels. It is a useful structured-output neighbor because validity improves when the model names the editable objects directly.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="LottieGPT summary">
   <div><strong>Problem</strong><span>Vector animation is structured JSON with hierarchy, timing, paths, properties, and easing behavior.</span></div>
   <div><strong>Mechanism</strong><span>Add Lottie-specific tokens, compress static and dynamic attributes, and train a multimodal autoregressive model in two stages.</span></div>
   <div><strong>Takeaway</strong><span>Validity improves when the grammar exposes the actual editable structure instead of only rendered frames.</span></div>
   <div><strong>Transfer boundary</strong><span>Lottie grammar is not map topology, but the discipline of grammar-first tokenization transfers directly.</span></div>
</div>

[LottieGPT](https://arxiv.org/abs/2604.11792) is not about aerial imagery, but it is deeply relevant to structured output. It tokenizes the objects that matter to an animator: layers, shapes, properties, keyframes, and easing curves.

The map-vector version is to tokenize what matters to an editor: rings, holes, shared edges, class boundaries, polygon closure, and maybe confidence or repair markers. If the output language does not name those entities, the model has to reinvent them implicitly.

## Ordering And Compute

### FVAR: next-focus prediction instead of next-scale prediction

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/fvar/fvar-next-focus.png" alt="FVAR diagram showing a progressive refocusing pyramid for visual autoregressive modeling">
   <figcaption>FVAR changes the autoregressive schedule: the next target is a sharper focus state, not merely the next raster scale. The figure shows that "next token" is a policy decision, and the policy can be designed around the next useful state.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="FVAR summary">
   <div><strong>Problem</strong><span>Visual AR models often use a fixed next-scale schedule that can blur detail and waste work.</span></div>
   <div><strong>Mechanism</strong><span>Construct a progressive refocusing pyramid with defocus kernels, train a high-frequency residual teacher, then distill into a vanilla VAR-compatible student.</span></div>
   <div><strong>Takeaway</strong><span>Autoregressive order is a policy. It can target the next useful state rather than a rigid raster scale.</span></div>
   <div><strong>Transfer boundary</strong><span>The optics are image-specific; the transferable part is learned ordering and focus, not defocus math.</span></div>
</div>

[FVAR](https://arxiv.org/abs/2511.18838) matters because it asks what "next" should mean. The answer is not necessarily the next fixed scale. It can be the next focus state: less blurred, more detailed, and more recoverable.

For structured geometry, the analogous question is which unit should be decoded next. The model could choose the largest object, the most confident polygon, the next uncertain boundary, the next class layer, or the next topology defect. That is a research question, not a formatting detail.

### VibeToken: make token count a control knob

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/vibetoken/vibetoken-method.png" alt="VibeToken method diagram showing dynamic one-dimensional image tokenizer for variable resolution generation">
   <figcaption>VibeToken decouples output resolution from token length with dynamic positional embeddings, adaptive patching, and adaptive decoding. The figure matters because token count becomes a controllable quality/compute knob instead of a fixed side effect of image size.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="VibeToken summary">
   <div><strong>Problem</strong><span>High-resolution autoregressive image generation can become expensive because token count scales with resolution.</span></div>
   <div><strong>Mechanism</strong><span>Train dynamic-length 1D latents with dynamic grid positional embeddings, adaptive patch embedding, and adaptive decoder resolution.</span></div>
   <div><strong>Takeaway</strong><span>Token length can become a quality/compute setting instead of a fixed consequence of image size.</span></div>
   <div><strong>Transfer boundary</strong><span>Polygon complexity depends on scene content, not image resolution alone, so budget control needs geometry-aware metrics.</span></div>
</div>

[VibeToken](https://arxiv.org/abs/2604.24885) is a compute-control paper in tokenizer clothing. It makes the number of 1D image tokens explicit, then trains generation to work across variable token budgets and resolutions.

That is a surprisingly good fit for polygon work. A sparse suburban tile should not spend the same output budget as dense downtown imagery. A geometry decoder should be able to condition on a token budget, or learn to request additional tokens when the scene demands it.

## Topology And Scene Geometry

### ACPV-Net: topology by reconstruction, not by sequence alone

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/acpv-net/acpv-net-architecture.png" alt="ACPV-Net architecture diagram showing semantic conditioning, vertex heatmap prediction, and polygonal vector map reconstruction">
   <figcaption>ACPV-Net predicts semantic and vertex evidence, then reconstructs an all-class planar vector map with shared-edge consistency. The cropped architecture figure makes the key contract visible: dense semantic cues and diffusion-generated vertices feed a planar graph backend that enforces topology.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="ACPV-Net summary">
   <div><strong>Problem</strong><span>Real vector maps need all-class polygons that meet cleanly, share boundaries, and avoid gaps or overlaps.</span></div>
   <div><strong>Mechanism</strong><span>Predict semantic masks and diffusion-generated vertex heatmaps, then build a planar straight-line graph and trace faces.</span></div>
   <div><strong>Takeaway</strong><span>Topology can be a backend contract rather than something the decoder must learn implicitly.</span></div>
   <div><strong>Transfer boundary</strong><span>The method is not pure autoregressive generation, so comparisons should separate accuracy, topology, and pipeline complexity.</span></div>
</div>

[ACPV-Net](https://arxiv.org/abs/2603.16616) is the most direct geospatial neighbor in this batch. It targets all-class polygonal vectorization: a seamless planar map, not just detached building outlines. Its strongest claim is topological consistency by construction, including reported zero gap/overlap and 100 percent shared-edge consistency on Deventer-512 global metrics.

This is the philosophical foil for pure sequence decoding. A sequence approach is simple and expressive, but it must learn topology and stopping behavior. ACPV-Net predicts intermediate fields, then lets a graph algorithm enforce topology.

### PixARMesh: sequence generation for mesh-native scenes

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/pixarmesh/pixarmesh-pipeline.png" alt="PixARMesh pipeline showing single-view scene reconstruction as autoregressive pose and mesh token generation">
   <figcaption>PixARMesh emits object pose and mesh tokens in one autoregressive sequence from a single RGB image and scene context. The figure shows a broader structured-geometry grammar where pose, object identity, and mesh-native shape tokens share one sequence.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="PixARMesh summary">
   <div><strong>Problem</strong><span>Single-view scene reconstruction needs object layout and detailed geometry, not only depth or occupancy.</span></div>
   <div><strong>Mechanism</strong><span>Fuse depth-derived point clouds, image features, masks, and scene context, then decode pose tokens plus mesh tokens autoregressively.</span></div>
   <div><strong>Takeaway</strong><span>Pose and geometry can share one token sequence when normalized into a common grammar.</span></div>
   <div><strong>Transfer boundary</strong><span>Mesh reconstruction depends on perception quality, especially segmentation; the same bottleneck will appear in map vectorization.</span></div>
</div>

[PixARMesh](https://arxiv.org/abs/2603.05888) takes the Pix2Seq idea into 3D scenes. The model emits object pose and native mesh tokens, not just a raster or implicit field. That matters for this report because structured visual workflows often need geometry beyond flat footprints: rooms, planes, objects, dimensions, and material state.

A good tokenizer can carry layout, pose, topology, and shape in one grammar if the sequence is designed around the geometry rather than around a convenient file format.

## Sources

Paper anchors: [Pix2Seq](https://arxiv.org/abs/2109.10852), [Pix2Seq v2](https://arxiv.org/abs/2206.07669), [Pix2Poly](https://arxiv.org/abs/2412.07899), [AToken](https://arxiv.org/abs/2509.14476), [VectorArk](https://arxiv.org/abs/2605.24398), [LottieGPT](https://arxiv.org/abs/2604.11792), [FVAR](https://arxiv.org/abs/2511.18838), [VibeToken](https://arxiv.org/abs/2604.24885), [ACPV-Net](https://arxiv.org/abs/2603.16616), and [PixARMesh](https://arxiv.org/abs/2603.05888).