---
title: 'ClipGS: Clippable Gaussian Splatting'
subtitle: 'For Interactive Cinematic Visualization of Volumetric Medial Data'
author: 'Julian Bohnenkämper'
affiliation:
  - 'Medizinische Bild- und Signalverarbeitung'
  - '12.02.2026'
---

# Scenario

<div style="display: none;">
$\renewcommand{\vec}[1]{\boldsymbol{\mathbf{#1}}}$
$\renewcommand{\mat}[1]{\boldsymbol{\mathbf{#1}}}$
$\newcommand{\R}{\mathbb{R}}$
$\newcommand{\e}{\mathrm{e}}$
$\newcommand{\d}{\mathrm{d}}$
$\newcommand{\G}{\mathcal{G}}$
</div>

::: columns-1-1

<div class="column">
<span>Input</span>
![](images/mri-volume-data.png){ width=400px }
<span>Volumetric Medical Data</span>
</div>

<div class="column fragment">
Pathtraced Render
![](videos/thorax.webm){ .slim-video loop="" autoplay="" width=380px}

::: incremental
- View $\vec v_i$
- Clip Plane Depth $z_i$
- [Not Interactive]{ .red }
:::

</div>

:::

# Method Preview

![](images/method.png){ height=500px }

# 3D Gaussian Splatting (3DGS)

::: column

<canvas id="render-output" style="width: 800px; height: 500px; border-radius: 20px;" width="1200" height="750"></canvas>

<script type="module" src="js/bicycle-splat/bicycle-splat.js"></script>

[gsplat.js renderer](https://github.com/huggingface/gsplat.js){ .small }

:::

# Gaussian Parameters

[$$ \alpha \cdot \mathcal G (\vec x; \vec \mu, \vec s, \vec r) \cdot \vec c $$]{ .minus-margin-top }

<div style="display: flex; gap: 1em;">
<style>
    #gui-container {
        /*position: absolute;*/
        display: flex;
        flex-direction: column;
        font-size: 0.5em;
        /*padding: 1em;*/
        /*top: 0;*/
        /*left: 0;*/
    }
    #gui-container span {
        margin-top: 1em;
    }
</style>
<div id="gui-container"></div>
<canvas id="surface" style="width: 640px; height: 480px; border-radius: 20px;"></canvas>
</div>
<script type="module" src="js/one-splat/js/index.js"></script>

# 3DGS Training

:::columns-1-1

:::column
![](images/chair-views.png){ .rounded height=350px }
[Renders created with [PlenOctrees](https://alexyu.net/plenoctrees/)]{ .small }
:::

:::column
![](videos/3dgs-training.mp4){ .rounded controls="" height=350px }

[Lecture: Computer Graphics --- Mario Botsch](https://cg.cs.tu-dortmund.de/downloads/teaching/graphics/slides/25-splatting-deck.html#/d-splat-training-1){ .small }
:::

:::

<!-- 
[:vspace](0.5em)

$$ \mathcal L (\mat I, \mat I_{\text{gt}}) = (1 - \lambda) \mathcal L_1 (\mat I, \mat I_{\text{gt}}) + \lambda \mathcal L_{\text{D-SSIM}} (\mat I, \mat I_{\text{gt}}) $$ -->

# Base Method

![](images/base-method.png){ height=500px }

# Hard Truncation (HT)

![](images/base-clipping.png){ height=500px }

# Hard Truncation Method

![](images/hard-truncation-method.png){ height=500px }

# Adaptive Adjustment Model (AAM)

![](images/aam-idea.png){ height=350px }

::: columns-1-1

![](images/aam-mlp.png){ .fragment width=350px }

[$$ \begin{align*}\vec \mu' &= \vec \mu + \delta \vec \mu \\ \vec s' &= \vec s + \delta \vec s \\ \vec r' &= \vec r + \delta \vec r \end{align*}$$]{ .fragment }

:::

# Positional Encoding (PE)

[$$ \gamma(x) = [ \sin(2^0 \pi x), \cos(2^0 \pi x), \dots, \sin(2^L \pi x), \cos(2^L \pi x) ]^\top $$]{ .minus-margin-top }

<!--- TODO: Nice visualization of vector?-->

::: fragment

Example --- [NeRF](https://www.matthewtancik.com/nerf)

::: columns-1-1

::: column
![](images/with-pe-cropped.png){ height=350px }
With PE
:::

::: { .column .fragment }
![](images/no-pe-cropped.png){ height=350px }
Without PE
:::

:::

:::

# HT + AAM Method

![](images/ht-aam-method.png){ height=500px }

# Learned Truncation

::: columns-1-1

![](images/learned-truncation.png){ height=500px }

::: incremental
- $\cancel{(\vec \mu + \vec \delta) \cdot \vec n} < z$
- $m < z$
    - One degree of freedom necessary
    - Decouple clipping and position
- [Not differentiable]{ .red }

![](images/step-function-reversed.png){ .fragment width=270px }

:::

:::

# Straight Through Estimator

::: { .columns-1-1 }

::: column
Forward Pass
![](images/step-function-reversed.png){ width=350px }
:::

::: column
Backward Pass
![](images/smooth-step-function-reversed.png){ width=350px }
:::

:::

<!-- 
$$
\mathcal M = \text{sg}\left(\mathbb{1}[\sigma(z - m) > \epsilon] - \sigma(z - m)\right) + \sigma(z - m)
$$

::: { .columns-1-1 .absolute }

::: { .column .fragment }
![](images/Logistic-curve.png){ width=230px }
$\sigma(\:\cdot\:) \text{: sigmoid}$
:::

::: column
[$$\mathbb 1 [ \mathcal B ] = \begin{cases} 1, \text{ if } \mathcal B \text{ is true} \\ 0, \text{ if } \mathcal B \text{ is false} \end{cases}$$]{ .fragment }
[$$\text{sg}(\:\cdot\:) \text{: stop gradient}$$]{ .fragment .slight-minus-margin-top-this }
:::

::: -->

<!--[`vis_mask = ((torch.sigmoid(-dists) > 0.5).float() - torch.sigmoid(-dists)).detach() + torch.sigmoid(-dists)`]{ .small }-->

# Full Method

![](images/method.png){ height=500px }

# Methods for Comparison

<!--$z_i$ as time parameter for dynamic models-->

::: incremental
- HexPlane: A Fast Representation for Dynamic Scenes
- GauFRe: Gaussian Deformation Fields for Real-time Dynamic Novel View Synthesis
- N-DG: N-Dimensional Gaussians for Fitting of High Dimensional Functions
- 4DGaussians: 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering
:::

# Comparison --- Head

![](images/head-comparison.png){ height=500px }

<!-- 
# Comparison --- Lower Limb

![](images/lower-limb-comparison.png){ height=500px } -->

<!-- 
# Comparison --- Skull

![](images/skull-comparison.png){ height=500px } -->

# Comparison --- Thorax

![](images/thorax-comparison.png){ height=500px }

# Comparison --- Liver & Kidney

![](images/liver-and-kidney-comparison.png){ height=500px }

# Comparison --- Quantitative

![](images/quantitative-table.png){ width=1200px }

<!--
- FPS
    - Outlier: HexPlane
- Training Time
    - Outlier: N-DG
    - Every one else ~1 coffe drinking experience
- Storage
    - Outliers: HexPlane, (N-DG)-->

# Ablation Study

![](images/ablation-table.png){ width=1200px }

<!--
<div style="display: flex; gap: 0.5em">
``` { .bar-chart width="150px" height="500px" }
PSNR ↑
LT, 34.577
AAM, 35.946
AAM + HT, 36.255
AAM + LT, 36.635
```

``` { .bar-chart width="150px" height="500px" }
SSIM ↑
LT, 0.969
AAM, 0.971
AAM + HT, 0.971
AAM + LT, 0.974
```

``` { .bar-chart width="150px" height="500px" }
LPIPS ↓
LT, 0.070
AAM, 0.064
AAM + HT, 0.063
AAM + LT, 0.061
```

``` { .bar-chart width="150px" height="500px" }
Time [min] ↓
LT, 11.1
AAM, 11.5
AAM + HT, 14.1
AAM + LT, 13.7
```

``` { .bar-chart width="150px" height="500px" }
FPS ↑
LT, 201
AAM, 148
AAM + HT, 150
AAM + LT, 156
```

``` { .bar-chart width="150px" height="500px" }
Storage [MB] ↓
LT, 14.7
AAM, 15.6
AAM + HT, 15.7
AAM + LT, 16.1
```
</div>-->

# Summary & Outlook

::: incremental

- Interactive cinematic rendering of medical data
- Extention of 3DGS
    - Learned Truncation
    - Adaptive Adjustment Model
- Medical cinematic dataset
- Incremental improvement

:::

[:vspace](0.5em)

::: fragment
What's next?

::: incremental
- Clip plane direction?
- Even more natural clipping possible?
:::

:::