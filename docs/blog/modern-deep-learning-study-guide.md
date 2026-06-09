<a class="cvpr-back-link" href="../cvpr2026/">Back to CVPR 2026 index</a>

# A Personal Study Guide To Modern Deep Learning

This page condenses my CVPR 2026 study notes into one practical guide. It is not a survey of every paper. It is a way to rebuild the core ideas from small examples, then use the conference papers as anchors for the modern patterns that keep reappearing.

The rule is simple: if I cannot explain a mechanism with a toy tensor shape, a tiny diagram, and a plain-language example, I do not understand it well enough to reuse it.

<p class="cvpr-callout">The through-line is representation plus update rule: choose what the model stores, choose how information enters, choose what operation changes the state, and choose how validity is checked.</p>

<div class="study-mermaid-shell" aria-label="Modern deep learning mechanism map">
<div class="mermaid">
flowchart LR
    A[Input evidence] --> B[Representation]
    B --> C[Conditioning]
    C --> D[Update rule]
    D --> E[Validity check]
    E --> F[Useful output]
    B --> B1[Tokens]
    B --> B2[Latents]
    B --> B3[Geometry]
    D --> D1[Attention]
    D --> D2[Denoising]
    D --> D3[Flow]
    D --> D4[Draft and refine]
</div>
</div>

## How To Read This Page

Read it in order the first time. The page starts with a tiny vision transformer example and then uses the same few questions everywhere:

1. What is represented?
2. Where does conditioning enter?
3. What operation changes the state?
4. What keeps the output valid?

Those four questions are enough to understand many papers quickly. They work for object detection, polygon generation, diffusion, 3D reconstruction, visual-language models, and verifier-based inference.

## 1. A 224 x 224 Image Becomes Tokens

Start with the smallest concrete example.

```text
image = 224 x 224 x 3
patch size = 16 x 16
embedding dimension = 384
```

The image is split into patches:

```text
patches per side = 224 / 16 = 14
total patches = 14 x 14 = 196
one patch = 16 x 16 x 3 = 768 raw numbers
linear projection = 768 -> 384
patch tokens = [196, 384]
```

If the model adds one class token:

```text
full image sequence = [197, 384]
batch of 8 images = [8, 197, 384]
```

That is the first mental switch. A modern vision model usually does not process an image as one picture. It turns the image into a sequence of learned vectors.

<div class="study-mermaid-shell" aria-label="Image patching diagram">
<div class="mermaid">
flowchart LR
    A[224 x 224 x 3 image] --> B[14 x 14 grid]
    B --> C[196 patches]
    C --> D[flatten each patch: 768 numbers]
    D --> E[linear projection]
    E --> F[196 tokens, each 384 dims]
    F --> G[Transformer encoder]
</div>
</div>

### Attention Is Learned Lookup

For one attention head, imagine:

```text
X = [197, 384]
head dimension = 64

Q = X @ W_Q = [197, 64]
K = X @ W_K = [197, 64]
V = X @ W_V = [197, 64]
```

Then:

```text
attention scores = Q @ K^T = [197, 197]
output = softmax(scores / sqrt(64)) @ V = [197, 64]
```

Plainly:

- `Q` asks what a token is looking for.
- `K` says what each token contains.
- `V` is the information passed along if that token is selected.

Self-attention lets every image patch ask every other patch for useful context. This is why the corner of a roof can use evidence from the rest of the building.

### Cross-Attention Adds A Decoder

Now put a decoder on top. The image encoder gives memory:

```text
image memory = [197, 384]
partial output = [12, 384]
```

In cross-attention:

```text
Q comes from the output tokens: [12, 64]
K,V come from the image memory: [197, 64]
attention map = [12, 197]
```

The decoder tokens ask the image: where is the evidence for my next step?

This one shape explains many systems:

- DETR object queries ask image memory for objects.
- Pix2Seq-style decoders ask image memory for boxes, masks, or coordinates.
- Aerial2Poly asks image memory for the next coordinate, separator, class token, or stop token.
- VLM bridge modules ask image memory for the visual evidence a language model can consume.

## 2. Outputs Can Be A Language

The next modern idea is that the output does not have to be a mask or a class label. It can be a sequence.

For a tiny building rectangle in a `224 x 224` image:

```text
corners:
(56, 72), (132, 72), (132, 140), (56, 140)

tokens:
BOS CLASS_BUILDING X_056 Y_072 SEP X_132 Y_072 SEP
X_132 Y_140 SEP X_056 Y_140 EOS
```

This is why Pix2Seq, Pix2Seq v2, Pix2Poly, VectorArk, LottieGPT, ACPV-Net, and Aerial2Poly all belong in the same mental neighborhood. They ask: what is the right alphabet for visual structure?

<div class="paper-summary-grid study-summary-grid" aria-label="Structured output summary">
   <div><strong>Representation</strong><span>Boxes, polygons, vectors, meshes, actions, or scene states become tokens or structured latents.</span></div>
   <div><strong>Conditioning</strong><span>The image, a class prompt, a text prompt, or a task token tells the decoder what to produce.</span></div>
   <div><strong>Update rule</strong><span>The model writes left-to-right, fills masked tokens, refines a draft, or reconstructs a graph.</span></div>
   <div><strong>Validity</strong><span>A parser, verifier, topology backend, or metric checks whether the output is useful, not only likely.</span></div>
</div>

The alphabet is not clerical. It decides what errors are easy or hard.

```text
plain coordinates:     X_056 Y_072 X_132 Y_072 ...
richer grammar:        START_RING X_056 Y_072 EDGE_RIGHT EDGE_DOWN ...
topology backend:      semantic field + vertices -> planar graph -> polygons
```

[AToken](https://arxiv.org/abs/2509.14476) asks whether images, videos, and 3D can share a visual alphabet. [VectorArk](https://arxiv.org/abs/2605.24398) asks which vector primitives make image vectorization stable. [LottieGPT](https://arxiv.org/abs/2604.11792) shows that vector animation becomes easier when the model predicts the editable hierarchy instead of rendered frames. [ACPV-Net](https://arxiv.org/abs/2603.16616) shows the opposite pressure: sometimes topology should be enforced by a graph backend rather than left entirely to a sequence model.

The lesson for modern deep learning is broad: before choosing a bigger model, choose the language the model must speak.

## 3. Detection Queries Are Empty Output Slots

DETR-style detection is a second foundation pattern. Instead of scanning anchors, the model learns output slots.

```text
image -> encoder memory = [197, 384]
object queries = [100, 384]
decoder output = [100, 384]
each output -> class + box or no-object
```

Plainly: the model gets 100 empty cards and decides which cards should become objects.

Deformable attention makes this cheaper and more focused. Suppose a multi-scale image memory has:

```text
1/8 feature map  = 28 x 28 = 784 locations
1/16 feature map = 14 x 14 = 196 locations
1/32 feature map = 7 x 7 = 49 locations
total = 1029 locations
```

Dense cross-attention from 100 queries costs:

```text
100 x 1029 = 102,900 query-location comparisons
```

If each query samples 4 points per level over 3 levels:

```text
100 x 3 x 4 = 1,200 sampled points
```

The model no longer reads the whole image equally. Each query learns where to look.

<div class="study-mermaid-shell" aria-label="Dense attention versus deformable attention">
<div class="mermaid">
flowchart TB
    Q[100 object queries]
    M[1029 image locations]
    S[12 sampled locations per query]
    Q -->|dense attention| M
    Q -->|deformable attention| S
    M --> O[class and box slots]
    S --> O
</div>
</div>

This is the same family of ideas as active perception and adaptive crop retrieval. AwaRes, AdaptVision, ZoomEarth, and GeoViS all say some version of: first get the global context, then spend high-resolution attention where it matters.

## 4. Diffusion Means Repair A Corrupted State

The useful view of diffusion is not "make pretty images." It is:

```text
corrupt a clean state
train a model to repair it
sample by repeatedly repairing
```

The standard Gaussian noising equation is:

<div class="cvpr-math-box" markdown="1">
$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon
$$

Here, `x_0` is clean data, `epsilon` is random noise, and `x_t` is the corrupted version at time `t`. The denoiser receives `x_t`, the timestep, and optional conditioning, then predicts something useful: noise, the clean sample, a score, or a velocity.
</div>

Tiny scalar example:

```text
clean value x_0 = 10
keep 70 percent signal, add 30 percent noise
noise epsilon = -2

x_t = 0.7 * 10 + 0.3 * (-2) = 6.4
```

The model sees `6.4` plus context and learns how to move it back toward `10`.

For a polygon, the corrupted state might be:

```text
clean:     CLASS X_056 Y_072 SEP X_132 Y_072 SEP X_132 Y_140 EOS
corrupted: CLASS X_056 MASK  SEP MASK  Y_072 SEP X_132 Y_140 EOS
repair:    recover missing coordinate tokens and keep syntax valid
```

<div class="study-mermaid-shell" aria-label="Diffusion and flow repair diagram">
<div class="mermaid">
flowchart LR
    A[Clean geometry] -->|corrupt| B[Damaged geometry]
    B -->|image-conditioned denoiser| C[Repaired geometry]
    D[Noise latent] -->|flow field| E[Cleaner latent]
    E -->|decode| C
</div>
</div>

The CVPR diffusion and flow papers become easier through this lens:

- [Mean Flows](https://arxiv.org/abs/2505.13447) and [Improved Mean Flows](https://arxiv.org/abs/2512.02012) ask whether a model can learn a strong average motion from noisy state to clean state.
- [ELF](https://arxiv.org/abs/2605.10938) does the repair in token-embedding space, then discretizes at the end.
- [Back to Basics](https://arxiv.org/abs/2511.13720) says the target should often be the clean object itself.
- [Duo](https://arxiv.org/abs/2506.10892) and [Duo++](https://arxiv.org/abs/2602.21185) make discrete tokens revisable.
- [DiDiCM](https://arxiv.org/abs/2511.20263) shows diffusion-style refinement even for classification decisions.
- [Visual Diffusion Models are Geometric Solvers](https://arxiv.org/abs/2510.21697) makes the repair-as-constraint-solving view explicit.

The transfer question is not whether every structured-output model should become diffusion. The question is narrower and more useful: which errors should remain editable after the first draft?

## 5. Flow Means Learn The Wind Field

Diffusion says: denoise step by step.

Flow matching says: learn the velocity field that moves noisy samples toward data.

```text
start: random point
model predicts velocity
take a step
repeat until the point looks like data
```

The simplest equation is:

<div class="cvpr-math-box" markdown="1">
$$
\frac{dx}{dt} = v_t(x, c)
$$

`x` is the current state, `t` is time, `c` is conditioning, and `v_t` is the learned direction of motion. In words: where should this state move next?
</div>

For an image model, `x` may be a latent image grid. For a geometry model, `x` could be a continuous embedding of a polygon draft. This is why embedding-space flow papers matter: they separate smooth movement from final discrete output.

Mean-flow-style models are attractive because they try to reduce the number of function calls. The warning is equally important: a one-step model is only as good as its target. Jumping faster is not useful if the jump is poorly defined.

## 6. 3D Starts With One Pixel And One Depth

Modern 3D vision looks intimidating until the smallest example is clear.

Given a camera with:

```text
fx = fy = 800
cx = 112
cy = 112
```

and a pixel:

```text
u = 144
v = 112
depth z = 8 meters
```

Back-project into 3D:

```text
x = (u - cx) * z / fx = (144 - 112) * 8 / 800 = 0.32
y = (v - cy) * z / fy = (112 - 112) * 8 / 800 = 0.00
z = 8.00
```

A pixel alone is a ray. A pixel plus depth is a point.

<div class="study-mermaid-shell" aria-label="Pixel to 3D diagram">
<div class="mermaid">
flowchart LR
    A[Image pixel u,v] --> B[Camera intrinsics]
    A --> C[Depth z]
    B --> D[Back-projection]
    C --> D
    D --> E[3D point x,y,z]
    E --> F[Point map, depth map, track, or scene token]
</div>
</div>

From there, most 3D representations are different storage choices:

| Representation | Plain meaning | Good for | Hard part |
| --- | --- | --- | --- |
| Point cloud | list of 3D points | simple geometry | no surfaces |
| Mesh | vertices plus faces | explicit surfaces | topology |
| Voxel grid | 3D cells | CNN-like processing | memory |
| NeRF | neural function queried along rays | novel views | slow rendering |
| 3D Gaussian splats | soft ellipsoids in space | fast rendering | pruning and semantics |
| Scene tokens | learned summary vectors | foundation models | interpretability |

[VGGT-Omega](https://arxiv.org/abs/2605.15195) is important because it treats reconstruction as a route to a reusable geometry backbone. Camera tokens and scene/register tokens compress cross-frame spatial consistency. [SAM 3D](https://arxiv.org/abs/2511.16624) is important because its data engine is a mechanism: generate candidates, have humans verify or rank them, then use those preferences to improve the system. [D4RT](https://arxiv.org/abs/2512.08924) shows a query interface over dynamic 4D scene memory. [TokenGS](https://arxiv.org/abs/2604.15239) decouples Gaussian prediction from pixels with learnable scene tokens.

For property and geospatial work, the lesson is not "everything must become 3D." It is that an image can be a projection of a richer scene state. Footprints, height, roof facets, material, camera pose, and condition evidence are different views of that state.

## 7. Visual-Language Models Need Evidence, Not Just Words

A common VLM shape is:

```text
image -> vision encoder -> visual features
visual features -> adapter/query tokens -> language model
text + visual tokens -> answer
```

Toy shapes:

```text
image tokens = [197, 1024]
query tokens = [32, 768]
language tokens = [N, 768]
LLM input = [N + 32, 768]
```

The adapter is a translator. It turns visual features into tokens the language model can use.

This framing exposes the failure modes:

- the visual evidence may be too low-resolution;
- the adapter may compress away small details;
- the language model may rely on priors instead of pixels;
- the context may be too long and poorly organized;
- the benchmark may not test the failure that matters.

This is why [AdaptVision](https://arxiv.org/abs/2512.03794), [AwaRes](https://arxiv.org/abs/2603.16932), and [ZoomEarth](https://arxiv.org/abs/2511.12267) are conceptually important. They do not process every pixel at high resolution by default. They first inspect a cheaper overview, then request high-resolution crops or regions when the question needs detail.

<div class="study-mermaid-shell" aria-label="Adaptive visual evidence diagram">
<div class="mermaid">
flowchart LR
    A[Low-resolution overview] --> B{Enough evidence?}
    B -->|yes| C[Answer]
    B -->|no| D[Request crop]
    D --> E[High-resolution evidence]
    E --> C
</div>
</div>

Other VLM reliability papers teach the same systems lesson:

- OddGridBench shows that tiny visual differences are not tiny for model reliability.
- SceneBench and Scene-RAG organize long-video evidence around scenes.
- SceneScribe-1M and MAVEN turn video data into structured semantic, geometric, temporal, and event annotations.
- ENC-Bench and GeoMMBench show that maps and geospatial symbols require domain scaffolding.
- Material-classification work shows that frozen foundation priors can help, but only after a domain-specific data and feature pipeline makes the material evidence visible.

The practical rule: if the model needs exact evidence, make evidence acquisition an explicit part of the architecture.

## 8. Test-Time Scaling Means Spend Inference Compute Wisely

Training-time improvement changes the model before deployment. Test-time scaling spends more compute during inference.

```text
one-shot:        model -> answer
test-time:       model -> candidates -> verifier -> refined answer
```

This is already familiar from human work. We write drafts, check them, revise them, and only then publish.

<div class="study-mermaid-shell" aria-label="Draft verify refine diagram">
<div class="mermaid">
flowchart LR
    A[Input] --> B[Cheap draft]
    B --> C[Verifier]
    C --> D{Good enough?}
    D -->|yes| E[Final output]
    D -->|no| F[Refiner]
    F --> C
</div>
</div>

For a vector map, the verifier can score:

- boundary alignment;
- closure;
- self-intersections;
- duplicate polygons;
- missing foreground;
- topology gaps and overlaps;
- class plausibility;
- downstream measurement error.

Thinking with Drafts, verifier-guided decoding, adaptive crop retrieval, and discrete denoising all fit into this general shape. The model does not have to be perfect in one pass if the system can make a good draft and then repair the specific mistakes that matter.

## 9. A Personal Concept Map

The table below is the shortest version of the guide. Each row is a modern deep learning concept, explained as a reusable mechanism.

| Concept | Small example | What changes | What to ask when reading a paper |
| --- | --- | --- | --- |
| Patch tokens | `224 x 224` image -> 196 tokens | pixels become a sequence | what information did tokenization preserve or lose? |
| Self-attention | every patch queries every other patch | context flows across tokens | what evidence can each token access? |
| Cross-attention | output tokens query image memory | decoder uses visual evidence | where does the condition enter? |
| Object queries | 100 empty slots become boxes | output slots compete for objects | what does a slot represent? |
| Geometry tokens | coordinates/classes become a sentence | structure becomes language | is the grammar parseable and valid? |
| Diffusion | corrupted state gets repaired | iterative denoising | what is the clean target? |
| Flow | state follows a velocity field | continuous transport | how many evaluations are needed? |
| Discrete diffusion | token sequence is corrupted and restored | earlier tokens stay editable | which mistakes should remain revisable? |
| Scene tokens | multiple views produce compact geometry memory | images become reusable 3D state | what geometry is explicit? |
| Adaptive evidence | overview first, high-res crop only when needed | compute follows uncertainty | what tells the model to look closer? |
| Draft/refine | cheap attempt then verifier repair | inference becomes a loop | what does the verifier actually measure? |

## 10. Reading Path

If I were relearning this from scratch, I would read in this order:

1. Vision transformers and attention: learn the `224 x 224 -> patches -> tokens` shape.
2. DETR and cross-attention: learn output slots and image-memory querying.
3. Pix2Seq lineage: learn how visual outputs become token languages.
4. Diffusion and flow: learn generation as repair or transport.
5. Discrete diffusion and draft/refine: learn why tokens should remain editable.
6. VGGT-Omega, SAM 3D, and Gaussian splatting: learn scene representation beyond pixels.
7. AToken, VectorArk, FVAR, and ACPV-Net: learn alphabet, order policy, token budget, and topology.
8. AdaptVision, ZoomEarth, OddGridBench, SceneBench, and MAVEN: learn why reliable visual systems need evidence acquisition, memory, data engines, and evaluation.

## Paper Anchors

This guide is grounded in my CVPR 2026 reading and public paper sources, especially [AToken](https://arxiv.org/abs/2509.14476), [VectorArk](https://arxiv.org/abs/2605.24398), [FVAR](https://arxiv.org/abs/2511.18838), [ACPV-Net](https://arxiv.org/abs/2603.16616), [Mean Flows](https://arxiv.org/abs/2505.13447), [Improved Mean Flows](https://arxiv.org/abs/2512.02012), [ELF](https://arxiv.org/abs/2605.10938), [Back to Basics](https://arxiv.org/abs/2511.13720), [Duo](https://arxiv.org/abs/2506.10892), [Duo++](https://arxiv.org/abs/2602.21185), [VGGT-Omega](https://arxiv.org/abs/2605.15195), [SAM 3D](https://arxiv.org/abs/2511.16624), [AdaptVision](https://arxiv.org/abs/2512.03794), [ZoomEarth](https://arxiv.org/abs/2511.12267), [OddGridBench](https://arxiv.org/abs/2603.09326), [SceneScribe-1M](https://arxiv.org/abs/2604.07990), and [MAVEN](https://arxiv.org/abs/2605.21917).