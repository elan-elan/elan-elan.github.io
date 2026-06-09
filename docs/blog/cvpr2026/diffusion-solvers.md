<a class="cvpr-back-link" href="../">Back to CVPR 2026 index</a>

# Diffusion, Mean Flow, And Training-Free Solvers

The useful diffusion lesson from CVPR 2026 is not just better image synthesis. It is a broader view of generation as repair: start from a noisy, incomplete, or corrupted state, then move it toward a valid output.

For structured outputs, that changes the question. Instead of asking whether one decoder must emit every token left-to-right, we can ask which parts should be drafted, corrupted, denoised, transported, verified, or repaired.

<div class="cvpr-flow" aria-label="Structured refinement flow">
   <span>corrupted draft</span>
   <span>denoising or flow step</span>
   <span>constraint-aware repair</span>
   <span>valid structure</span>
</div>

## Mechanism Map

<div class="mechanism-map" aria-label="Diffusion and flow mechanism map">
   <div><strong>Direct transport</strong>Mean-flow and Bidirectional Normalizing Flow models ask the network to learn a large useful motion, not only tiny reverse steps.</div>
   <div><strong>Clean targets</strong>Back to Basics and ELF make the recoverable object explicit: clean pixels, embeddings, or tokens.</div>
   <div><strong>Revisable tokens</strong>Duo, Duo++, and DiDiCM keep discrete predictions editable instead of locking in every earlier choice.</div>
   <div><strong>Detail without bottlenecks</strong>PixelDiT separates global patch semantics from local pixel detail without committing to a VAE latent.</div>
   <div><strong>Adaptive compute</strong>SeaCache and DDiT spend denoising work where the state is still changing.</div>
   <div><strong>Geometry solving</strong>Training-free and visual-solver papers show diffusion as an update rule under constraints, not only a generator.</div>
</div>

The common pattern is:

```text
damaged state + condition -> repaired state
```

The damaged state can be an image, an embedding sequence, a discrete token canvas, a patch set from one image, or a rasterized geometry problem. That is why these papers matter for structured geometry: they make correction a first-class operation.

The easiest example is a damaged blueprint. If a corner is shifted, a separator is missing, or the drawing stops too early, a repair model should use the image and the current draft to fix the specific damage instead of redrawing everything from scratch.

```text
clean tokens:     CLASS x1 y1 x2 y2 x3 y3 SEP EOS
corrupted tokens: CLASS x1 y1 MASK y2 x3 y3 SEP EOS
repair target:    CLASS x1 y1 x2   y2 x3 y3 SEP EOS
```

That is the lens for the page: what is corrupted, what evidence conditions the repair, how big is the update, and how do we know the final structure is valid?

## A Small Toy

This toy is deliberately simple: the same target spiral can be approached as forward corruption, flow-like transport, or discrete token denoising. The point is not the spiral. The point is that the output can be a state that gets repaired.

<div id="spiral-diffusion-demo" class="diffusion-toy-demo"></div>
<script src="../../../assets/cvpr2026/diffusion-toys/spiral-demo.js"></script>

## Intuitive Math Box

<div class="cvpr-math-box" markdown="1">
**Three related views**

$$
\begin{aligned}
\mathrm{forward\ Gaussian:}\quad x_t &= \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon \\
\mathrm{denoising\ target:}\quad f_{\theta}(x_t, t, c) &\rightarrow x_0 \;\mathrm{or}\; \epsilon \\
\mathrm{conditional\ score:}\quad \nabla_x \log p_t(x \mid c) &= \nabla_x \log p_t(x) + \nabla_x \log p_t(c \mid x) \\
\mathrm{flow\ matching:}\quad \frac{dx}{dt} &= v_t(x, c) \\
\mathrm{mean\ flow\ view:}\quad x_{\mathrm{noise}} &\rightarrow x_{\mathrm{data}} \;\mathrm{with\ an\ average\ velocity}
\end{aligned}
$$

The forward noising equation can be written in closed form, but the useful inverse is usually not a closed-form solution. Conditioning is the practical lever: image evidence, text, masks, measurements, or verifier feedback change which clean state is plausible. For structured output, replace \(x\) with coordinate, class, separator, and EOS tokens, or with continuous embeddings for those tokens. The core question becomes: which corrupted state should the model learn to repair?
</div>

## Paper-by-paper takeaways

### Mean Flows: learn the average motion

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/mean-flows/mean-flows-overview.png" alt="Mean Flows diagram showing average velocity as the target for one-step generative modeling">
   <figcaption>Mean Flows replace a long sequence of reverse steps with an average-velocity target that can support one-step or few-step generation. The figure matters because it turns sampling into a direct transport problem: predict the useful displacement, not every tiny intermediate move.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="Mean Flows summary">
   <div><strong>Problem</strong><span>Diffusion samplers often need many small reverse steps, while one-step distillation can lose the geometry of the path.</span></div>
   <div><strong>Mechanism</strong><span>Train an average velocity field over an interval, then sample by applying the learned displacement from noise toward data.</span></div>
   <div><strong>Takeaway</strong><span>A strong repair map can be learned directly if the target matches the large move the sampler must execute.</span></div>
   <div><strong>Transfer boundary</strong><span>The paper is image-generation oriented; structured-output transfer needs a defined corrupted state and a clean target.</span></div>
</div>

[Mean Flows](https://arxiv.org/abs/2505.13447) is useful because it makes a blunt proposal: if a generative path is a motion from noise to data, train the model for the average motion over that interval. The result is not just a faster sampler. It is a different way to think about repair: predict the move that takes the current state closer to validity.

That suggests a narrow structured-output experiment: create damaged states with jitter, missing delimiters, swapped class tokens, or early stop tokens, then train a model to predict the clean state in one pass. The important thing is to train for the actual correction size, not a tiny step that will later be extrapolated.

### Improved Mean Flows: one-step maps still need stable targets

<figure class="paper-figure">
   <div class="paper-figure-pair">
      <img src="../../../assets/cvpr2026/papers/improved-mean-flows/improved-mean-flows-architecture.png" alt="Improved Mean Flows architecture with prediction tokens and a transformer backbone">
      <img src="../../../assets/cvpr2026/papers/improved-mean-flows/improved-mean-flows-v-loss.png" alt="Improved Mean Flows diagram reformulating MeanFlow as a velocity loss">
   </div>
   <figcaption>Left: Improved Mean Flows turns each condition into tokens and feeds them through the transformer with image latents. Right: the objective view explains why the paper changes the training target, reframing MeanFlow as a cleaner velocity-loss problem.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="Improved Mean Flows summary">
   <div><strong>Problem</strong><span>The original mean-flow objective can be self-referential and hard to optimize at scale.</span></div>
   <div><strong>Mechanism</strong><span>Reformulate the loss around a cleaner velocity target, use JVP structure carefully, and condition on guidance scale and context tokens.</span></div>
   <div><strong>Takeaway</strong><span>Fast-forward generation is not magic; the target and conditioning interface decide whether the jump is learnable.</span></div>
   <div><strong>Transfer boundary</strong><span>Flexible CFG and image-generation metrics do not translate directly to polygon decoding, but stable clean-target training does.</span></div>
</div>

[Improved Mean Flows](https://arxiv.org/abs/2512.02012) is the practical sequel. It keeps the idea of learning a large correction step, then adjusts the training objective and conditioning so the model has a clearer regression target. The paper also treats classifier-free guidance scale as an input, making guidance a controllable variable rather than a fixed afterthought.

For polygon work, the lesson is less about guidance and more about target design. A refiner should know whether it is repairing coordinates, delimiters, topology, or object inventory. A vague residual target will hide those distinctions.

### Bidirectional Normalizing Flow: from data to noise and back

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/biflow/biflow-reverse.png" alt="BiFlow reverse model diagram showing learned noise-to-data generation separate from the forward path">
   <figcaption>Bidirectional Normalizing Flow, abbreviated BiFlow by the authors, decouples the forward data-to-noise path from a separately learned noise-to-data reverse path. The diagram makes the key asymmetry visible: corruption can be easy to define while inversion still needs its own learned model.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="Bidirectional Normalizing Flow summary">
   <div><strong>Problem</strong><span>Classic normalizing flows tie generation to the exact inverse of the forward map, which can constrain the reverse direction.</span></div>
   <div><strong>Mechanism</strong><span>Train a data-to-noise forward model and a separate noise-to-data reverse model with hidden alignment and denoising supervision.</span></div>
   <div><strong>Takeaway</strong><span>If corruption is easy but inversion is hard, make inversion its own learned problem.</span></div>
   <div><strong>Transfer boundary</strong><span>The two-map setup still depends on a meaningful corruption path and can introduce reverse-model bias.</span></div>
</div>

[Bidirectional Normalizing Flow (BiFlow)](https://arxiv.org/abs/2512.10953) breaks a habit inherited from classic flows. The forward map can remain a good data-to-noise model, while the reverse path becomes a separate learned generator. That matters because the reverse direction is often where the real intelligence lives.

For structured labels, corruption is easy to define: jitter values, drop elements, flip delimiters, add tail noise after a stop token, or perturb ordering. The inverse is hard because it must use evidence and constraints. BiFlow says that inverse deserves its own model, not an analytic shortcut.

### ELF: flow in embedding space, discretize at the end

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/elf/elf-training-inference.png" alt="Embedded Language Flows training and sampling pipeline over continuous token embeddings">
   <figcaption>ELF keeps the denoising trajectory continuous in token-embedding space and maps back to discrete tokens only at the final unembedding step. The figure matters because it separates smooth repair from the final parseable vocabulary commitment.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="ELF summary">
   <div><strong>Problem</strong><span>Text and structured outputs are discrete, but many diffusion and flow tools are smoother in continuous spaces.</span></div>
   <div><strong>Mechanism</strong><span>Embed tokens, corrupt and denoise the continuous embeddings with a flow-style model, then unembed the final clean state into tokens.</span></div>
   <div><strong>Takeaway</strong><span>Continuous repair and discrete parseability do not have to be opposites if discretization is postponed.</span></div>
   <div><strong>Transfer boundary</strong><span>The final token projection is still a bottleneck; polygon syntax must remain recoverable after continuous refinement.</span></div>
</div>

[ELF](https://arxiv.org/abs/2605.10938) is the cleanest bridge from diffusion over images to diffusion over structured tokens. It does not ask a model to denoise hard token IDs directly. It flows through an embedding space, then commits back to vocabulary tokens at the end.

That is exactly the kind of middle ground structured geometry needs. Coordinate bins, class markers, separator tokens, and EOS are discrete because they must be parsed. But correction may be smoother if it happens in a continuous geometry embedding before the model snaps back to bins and grammar.

### Back To Basics: let the model predict the clean object

<figure class="paper-figure">
   <div class="paper-figure-pair">
      <img src="../../../assets/cvpr2026/papers/back-to-basics/back-to-basics-framework.png" alt="Back to Basics framework showing clean x-prediction for denoising generative models">
      <img src="../../../assets/cvpr2026/papers/back-to-basics/back-to-basics-x-pred-teaser.png" alt="Back to Basics teaser showing x prediction on the image manifold compared with epsilon and velocity targets">
   </div>
   <figcaption>Left: the model predicts the clean image patches directly. Right: the paper's intuition is that clean-object prediction aims back at the image manifold, while noise and velocity targets can point through harder auxiliary spaces.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="Back to Basics summary">
   <div><strong>Problem</strong><span>Noise and velocity prediction can become brittle when the target object is high-dimensional but structured.</span></div>
   <div><strong>Mechanism</strong><span>Return to clean-data prediction for denoising, especially in large patch spaces where the clean manifold is simpler than the noise target.</span></div>
   <div><strong>Takeaway</strong><span>Before adding sampler machinery, define the clean thing the denoiser should recover.</span></div>
   <div><strong>Transfer boundary</strong><span>The claim must be retested for polygon tokens, but the objective-design warning is directly relevant.</span></div>
</div>

[Back to Basics](https://arxiv.org/abs/2511.13720) is the useful antidote to overcomplicated diffusion thinking. If the data object is high-dimensional but lies on a structured manifold, predicting the clean object can be easier than predicting noise or velocity.

For polygons, the clean object is not a vague residual. It is a valid sequence or geometry tensor: class tokens in the right places, coordinate parity intact, separators present, EOS meaningful, vertices ordered, and contours non-self-intersecting. The repair model should be trained against that object.

### Efficient and Training-Free Single-Image Diffusion: structure can replace training only under strong assumptions

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/training-free-single-image-diffusion/training-free-method.png" alt="Training-free single-image diffusion method using patch extraction, Bayes denoising, reconstruction, and coarse-to-fine guidance">
   <figcaption>The training-free method replaces a neural denoiser with a closed-form patch-set denoiser built from a single image. It is useful here because it shows when analytic structure can substitute for training, and where those assumptions become narrow.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="Training-free single-image diffusion summary">
   <div><strong>Problem</strong><span>Can a diffusion-like generator work for one image without training a new denoising network?</span></div>
   <div><strong>Mechanism</strong><span>Build a patch dataset from the input image, denoise noisy patches by weighted Bayes averaging, then reconstruct coarse-to-fine.</span></div>
   <div><strong>Takeaway</strong><span>Training-free is credible only when the observation and corruption model provide enough structure.</span></div>
   <div><strong>Transfer boundary</strong><span>Most structured prediction needs cross-scene generalization; this is a boundary case, not a full replacement for learned conditional repair.</span></div>
</div>

[Efficient and Training-Free Single-Image Diffusion Models](https://arxiv.org/abs/2606.04299) is useful because it makes the assumptions visible. A single image can sometimes supply a local patch prior. If the corruption model is known, a closed-form denoiser can be built without training.

The analogous low-data idea is not training-free image generation. It is label-space repair: existing clean labels define structure, deliberate corruptions define the inverse problem, and evidence conditions the repair.

### Duo: discrete diffusion as token repair

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/duo/duo-schematic.png" alt="Duo schematic linking Gaussian diffusion to uniform-state discrete diffusion through an argmax bridge">
   <figcaption>Duo connects uniform-state discrete diffusion to Gaussian diffusion through an argmax bridge, making discrete tokens revisable. The schematic is the reason Duo belongs in this report: token outputs can be corrected rather than committed forever in left-to-right order.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="Duo summary">
   <div><strong>Problem</strong><span>Discrete diffusion over tokens can be slow and hard to train, especially with high-variance objectives.</span></div>
   <div><strong>Mechanism</strong><span>Use an argmax marginal bridge from Gaussian diffusion to uniform-state discrete diffusion, then train with lower-variance curriculum losses.</span></div>
   <div><strong>Takeaway</strong><span>The main attraction is revisability: a wrong token can be corrupted and corrected instead of fixed forever.</span></div>
   <div><strong>Transfer boundary</strong><span>Full random-token polygon generation is too high entropy as a first experiment; use conditional refinement after a draft.</span></div>
</div>

[Duo](https://arxiv.org/abs/2506.10892) matters because polygon outputs are token-like. Autoregressive decoding commits to a left-to-right history. Discrete diffusion keeps earlier tokens editable, which is exactly what malformed polygon sequences need.

The safe transfer is not to generate a whole scene from random tokens. Let the existing decoder draft the scene first. Then ask a Duo-style refiner to repair lower-entropy mistakes: coordinate jitter, missed separators, class-token swaps, and early EOS.

<figure class="paper-figure duo-comparison-figure">
   <div class="duo-gif-row" aria-label="Duo discrete diffusion compared with masked diffusion and autoregressive generation">
      <img src="../../../assets/cvpr2026/papers/duo-plus-plus/duo-2.gif" alt="Duo++ generation animation showing parallel token refinement">
      <img src="../../../assets/cvpr2026/papers/duo-plus-plus/mdlm-2.gif" alt="MDLM generation animation showing masked-token unmasking">
      <img src="../../../assets/cvpr2026/papers/duo-plus-plus/gpt-2.gif" alt="GPT-2 autoregressive generation animation showing left-to-right generation">
   </div>
   <figcaption>Duo++ refines many language tokens in parallel, MDLM-style masked diffusion gradually commits masked positions, and GPT-style autoregression moves left to right. The side-by-side animation makes the report's token-repair claim concrete: diffusion-style token models can revisit positions that an autoregressive decoder has already passed. GIFs are from the <a href="https://s-sahoo.com/duo-ch2/">Duo++ project page</a>, licensed CC BY-SA 4.0 by the authors.</figcaption>
</figure>

### Duo++: controlled corrector noise for discrete tokens

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/duo-plus-plus/duo-plus-plus-psi-samplers.png" alt="Duo++ Psi samplers showing controlled corrector noise for discrete diffusion">
   <figcaption>Duo++ introduces Psi-samplers that preserve the diffusion marginals while allowing controlled correction in discrete token space. The figure shows the practical sampler idea: a predictor step moves toward a clean token state, while a corrector step keeps an escape route from wrong confident tokens.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="Duo++ summary">
   <div><strong>Problem</strong><span>Discrete samplers need a way to revisit overconfident wrong tokens without exploding compute or memory.</span></div>
   <div><strong>Mechanism</strong><span>Psi-posteriors inject controlled corrector noise, while sparse top-k curriculum training avoids full-vocabulary cost.</span></div>
   <div><strong>Takeaway</strong><span>Correction is not only prediction; the sampler has to leave a route back from wrong confident states.</span></div>
   <div><strong>Transfer boundary</strong><span>Coordinate vocabularies and polygon grammar may need custom corruption schedules and parse-aware losses.</span></div>
</div>

[Duo++](https://arxiv.org/abs/2602.21185) is the practical extension. Its Psi-samplers are useful because they make self-correction explicit. The sampler can inject the right amount of uncertainty and still preserve the target diffusion marginals.

For polygon repair, this is a strong fit for coordinate bins and delimiters. A wrong corner should not force the entire rest of the sequence into a bad path. The refiner should be able to revisit it.

### DiDiCM: classification as discrete diffusion

<figure class="paper-figure">
   <div class="paper-figure-pair">
      <img src="../../../assets/cvpr2026/papers/didicm/didicm-main.png" alt="DiDiCM accuracy comparison against standard classifiers under different uncertainty settings">
      <img src="../../../assets/cvpr2026/papers/didicm/didicm-probability-evolution.png" alt="DiDiCM probability evolution over diffusion time for high-confidence, multiple-object, and ambiguous images">
   </div>
   <figcaption>Left: DiDiCM improves classification under uncertainty compared with a standard classifier baseline. Right: the probability traces show why the paper belongs here: the class belief is a state that gets sharpened over diffusion time, not a one-shot answer.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="DiDiCM summary">
   <div><strong>Problem</strong><span>Image classifiers usually emit a single posterior in one pass, leaving little room to represent and refine uncertainty.</span></div>
   <div><strong>Mechanism</strong><span>Model class prediction as discrete diffusion with a DiDiRN refinement network that iteratively sharpens a class distribution.</span></div>
   <div><strong>Takeaway</strong><span>The clever idea is that diffusion can refine decisions, not only pixels, tokens, or geometry states.</span></div>
   <div><strong>Transfer boundary</strong><span>A fixed class set is much easier than polygon coordinate grammar, so this is a conceptual bridge rather than a direct decoder recipe.</span></div>
</div>

[DiDiCM](https://arxiv.org/abs/2511.20263) deserves to be back in the report because it expands the diffusion analogy in exactly the right way. A class posterior can be treated as a state that is noisy, uncertain, and progressively refined. That is not the same as generating a footprint polygon, but it is a clean demonstration that diffusion-style thinking can apply to classification.

For Aerial2Poly, the transfer question is whether a structured output has intermediate decision states worth refining: object presence, class tokens, corner bins, separators, and EOS. DiDiCM says the answer can be yes even when the final output is a decision, not an image.

### 🏆 PixelDiT: global structure and local detail can share a pixel-space model

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/pixeldit/pixeldit-method.png" alt="PixelDiT architecture with patch-level and pixel-level transformer pathways">
   <figcaption>PixelDiT removes the fixed VAE bottleneck by combining a patch-level semantic stream with a pixel-level detail stream. The architecture is useful because it separates global layout from fine detail without permanently discarding high-frequency evidence.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="PixelDiT summary">
   <div><strong>Problem</strong><span>Latent diffusion is efficient, but a learned VAE bottleneck can lose local detail before the diffusion model sees it.</span></div>
   <div><strong>Mechanism</strong><span>Use patch-level tokens for global semantics, pixel-level tokens for detail, pixel-wise AdaLN, and temporary token compaction for efficiency.</span></div>
   <div><strong>Takeaway</strong><span>A bottleneck can be computational and reversible, not a permanent loss of detail.</span></div>
   <div><strong>Transfer boundary</strong><span>PixelDiT is large image generation machinery; the transferable part is two-level representation, not the full model.</span></div>
</div>

[PixelDiT](https://arxiv.org/abs/2511.20645) is a representation paper for this story. It separates global layout from local detail without hiding everything behind a fixed VAE latent.

The structured-output analogy is a two-level decoder: one stream for scene layout and object inventory, another for boundary and corner precision. The structure stream should not erase the detail stream before the model has a chance to recover precise vertices.

### 🏆 SeaCache: cache what is stable, recompute what is evolving

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/seacache/seacache-overview.png" alt="SeaCache overview showing spectral-evolution-aware cache reuse across diffusion steps">
   <figcaption>SeaCache aligns feature reuse with denoising spectral evolution rather than raw feature difference alone. The diagram shows that caching is a modeling decision: reuse features only when the frequency-aware denoising state says the content is stable.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="SeaCache summary">
   <div><strong>Problem</strong><span>Diffusion transformers repeat expensive computation, but naive cache reuse can ignore what is still changing.</span></div>
   <div><strong>Mechanism</strong><span>Estimate a cache distance filtered by expected denoising frequency response, then refresh only when meaningful content evolves.</span></div>
   <div><strong>Takeaway</strong><span>Reuse should be tied to the state of denoising, not just a fixed block-skip schedule.</span></div>
   <div><strong>Transfer boundary</strong><span>Natural-image spectral assumptions may not match aerial rasters or polygon token sequences.</span></div>
</div>

[SeaCache](https://arxiv.org/abs/2602.18993) is an efficiency paper with a modeling lesson. It asks whether the model is changing low-frequency layout or high-frequency detail before deciding to reuse cached features.

A structured refiner needs the same discipline. Cache stable spans. Recompute uncertain values, boundary-detail regions, or tokens that a verifier still dislikes. Do not spend equal compute on every token after the draft is already mostly correct.

### 🏆 ChordEdit: one-step repair needs a low-energy control field

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/chordedit/chordedit-method.png" alt="ChordEdit method diagram for one-step low-energy image editing with chord control">
   <figcaption>ChordEdit replaces a high-energy naive edit direction with a smoother one-step transport/control field. The figure matters because it shows why a one-step edit should preserve unchanged regions instead of dragging the whole sample through a crude source-to-target vector.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="ChordEdit summary">
   <div><strong>Problem</strong><span>One-step editing can distort non-edited regions when the model follows a naive source-to-target drift difference.</span></div>
   <div><strong>Mechanism</strong><span>Approximate a low-energy transport field, using a chord-style control direction to keep edits stable.</span></div>
   <div><strong>Takeaway</strong><span>A large repair step needs a smooth control prior, not only a raw difference vector.</span></div>
   <div><strong>Transfer boundary</strong><span>Polygon tokens are discrete unless the refiner works in continuous coordinates or embeddings.</span></div>
</div>

[ChordEdit](https://arxiv.org/abs/2602.19083) is not a polygon paper, but it is a good warning for one-step repair. Moving directly from source to target can churn regions that should stay fixed. A low-energy control field preserves more of the unchanged structure.

For structured editing, this means a correction model should not globally rewrite the sequence when only one local element is wrong. A smooth value-update or embedding-update field is a better prior than raw displacement everywhere.

### ★ DDiT: dynamic patch scheduling for denoising compute

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/ddit/ddit-overview.png" alt="DDiT overview comparing baseline diffusion transformer outputs with dynamic patch scheduling speedups">
   <figcaption>DDiT adapts patch scheduling across the denoising trajectory. The top row shows standard DiT using a fixed fine patch grid throughout; the bottom row shows DDiT spending coarser compute early and finer compute later, where the latent state still needs detail.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="DDiT summary">
   <div><strong>Problem</strong><span>Diffusion transformers spend similar patch-level compute even when early and late denoising steps have different needs.</span></div>
   <div><strong>Mechanism</strong><span>Use dynamic patch scheduling so coarse patches handle slower phases and finer patches handle faster latent evolution.</span></div>
   <div><strong>Takeaway</strong><span>The tokenization or patch size of the computation can change over the generation trajectory.</span></div>
   <div><strong>Transfer boundary</strong><span>Patch scheduling is orthogonal to polygon correctness; it helps after the representation and repair target are sound.</span></div>
</div>

[DDiT](https://arxiv.org/abs/2602.16968) belongs next to SeaCache. Both are about spending compute where it matters. DDiT changes patch granularity over time, using coarser patches when the state changes slowly and finer patches when detail matters.

This maps to refinement granularity. Early passes could repair coarse object layout or rough coordinates. Later passes could focus on high-curvature corners, uncertain boundaries, or image crops where the current structure disagrees with evidence.

### ★ Visual Diffusion Models are Geometric Solvers

<figure class="paper-figure">
   <img src="../../../assets/cvpr2026/papers/visual-diffusion-geometric-solvers/visual-diffusion-geometric-solvers-teaser.png" alt="Visual diffusion geometric solver examples with squares over curves">
   <figcaption>Visual Diffusion Models are Geometric Solvers frames diffusion as solving visualized geometry constraints, such as placing valid squares on curves. The figure matters because it treats the generated image as a candidate solution that can be snapped or checked by geometry, not just as a picture.</figcaption>
</figure>

<div class="paper-summary-grid" aria-label="Visual diffusion geometric solvers summary">
   <div><strong>Problem</strong><span>Can a visual diffusion model solve geometric constraint problems rather than merely generate plausible pictures?</span></div>
   <div><strong>Mechanism</strong><span>Condition on image-like problem instances, denoise solution visualizations, then snap or refine outputs toward exact geometry.</span></div>
   <div><strong>Takeaway</strong><span>Diffusion can be used as a visual constraint solver when the output state and verifier are explicit.</span></div>
   <div><strong>Transfer boundary</strong><span>Pixelized solver outputs still need vector snapping, topology checks, and exact geometric validation.</span></div>
</div>

[Visual Diffusion Models are Geometric Solvers](https://arxiv.org/abs/2510.21697) is the most literal geometry bridge here. It uses diffusion to solve tasks such as inscribed squares, Steiner trees, and polygon problems from visual inputs.

This suggests an alternate structured-output route: let diffusion propose or repair a rasterized geometry state, then snap it into vector form with a verifier. That is not a replacement for sequence decoding yet, but it is a legitimate way to think about diffusion as a solver over structure.

## Sources

Public anchors: [Mean Flows](https://arxiv.org/abs/2505.13447), [Improved Mean Flows](https://arxiv.org/abs/2512.02012), [Bidirectional Normalizing Flow (BiFlow)](https://arxiv.org/abs/2512.10953), [ELF](https://arxiv.org/abs/2605.10938), [Back to Basics](https://arxiv.org/abs/2511.13720), [Efficient and Training-Free Single-Image Diffusion Models](https://arxiv.org/abs/2606.04299), [Duo](https://arxiv.org/abs/2506.10892), [Duo++](https://arxiv.org/abs/2602.21185), [DiDiCM](https://arxiv.org/abs/2511.20263), [PixelDiT](https://arxiv.org/abs/2511.20645), [SeaCache](https://arxiv.org/abs/2602.18993), [ChordEdit](https://arxiv.org/abs/2602.19083), [DDiT](https://arxiv.org/abs/2602.16968), and [Visual Diffusion Models are Geometric Solvers](https://arxiv.org/abs/2510.21697). Duo++ project-page GIFs are from <https://s-sahoo.com/duo-ch2/> under the page's CC BY-SA 4.0 notice.