### Title - "GINI-informed and Metadata-conditioned variance priors for Hyperbolic Few-Shot Skin Lesion Classification"

### Contents
1. [The Problem Statement](#1-the-problem)
2. [Methodology](#2-methodology)
3. [Dataset](#3-dataset)
4. [Implementation Details](#4-implementation-details)
5. [Results](#5-results)


### 1. The Problem Statement
The task is to classify a skin lesion when each class is represented by only K labelled images (K = 3, 5 or 10), using the image and the patient's clinical record. With so few supports, a classifier can estimate where a class sits (its prototype) but not how widely it spreads, and a wrong spread estimate distorts every decision boundary. A nearest-prototype rule is wrong whenever classes differ in spread, and with limited support data, the spread is the thing we know least about.

**Why spread matters.** Skin lesion classes are not equally varied. For example, NEV in PAD-UFES-20 is a fairly uniform class, while SEK and ACK cover flat, raised, pigmented and keratotic presentations. A query sitting at the edge of a broad class can be closer to the center of a tight neighbour. A standard prototypical network assigns a query to the nearest class mean under Euclidean distance. Estimating a per-class, per-dimension variance from limited samples in a 64-dimensional embedding, is extremely noisy. 


### 2. Methodology

**Why Poincaré ball**
The labels form a tree: root, then benign and malignant, then six/seven diagnoses (and subtypes below those). In hyperbolic space the volume of a ball grows exponentially with its radius, as the number of nodes in a tree grows with depth. Trees therefore embed with very low distortion in a few hyperbolic dimensions, while Euclidean space needs many more dimensions for the same distortion. Superclasses can sit nearer the origin and fine classes towards the boundary. 

**Prototype charts.**
Why is variance measured after a Möbius shift? In raw ball coordinates the same coordinate step means a larger true distance near the boundary than near the centre. A variance computed there would mix real spread with position in the ball. We first move the prototype to the origin with the isometry h ↦ (−p) ⊕ h, then take the log map. In these coordinates 2‖ξ‖ equals the true hyperbolic distance. The variance is estimated and used in the same chart, so it means the same thing in both places. 

**The metadata-conditioned variance prior**
The prior answers a question the embedding cannot: given who these support patients are, how widely should lesions like theirs be expected to vary? 

Figure 1 illustrates the proposed Metadata-Conditioned Shrinkage Priors for Hyperbolic Prototypical Few-Shot Skin Lesion Classification model. 
![Figure 1. Model Architecture](model.png)

### 3. Dataset
Two publicly available skin lesion dataset namely PAD-UFES-20 and HAM10000 have been used. Each image is resized to 224×224 without cropping and encoded under six dihedral views: the identity, horizontal and vertical flips, and rotations by 90°, 180° and 270°. Because both backbones CLIP and DINOv2 are frozen, the features of all six views are computed once and cached. During training, each sampled image uses one view drawn uniformly at random; evaluation uses the identity view only.
For hierarchical structure of PAD-UFES-20 dataset, **benign**: ['NEV', 'SEK'], **malignant**: ['ACK', 'BCC', 'MEL', 'SCC'] is considered. ACK is pre-malignant and is considered malignant class. For HAM10000 dataset, **benign**: ['bkl', 'df', 'nv', 'vasc'], **malignant**: ['akiec', 'bcc', 'mel'] is considered.  Images are split 70:10:20 per class (train:validation:test) with no overlap. 


### 4. Implementation Details
The trainable model has 4.17 M parameters on top of two frozen backbones. Hyperparameters are given in the table below.

| Group | Setting | Value |
|---|---|---|
| Episodes | Ways / Shots / Queries per class | 6, 7 / 3, 5, 10 / 5 |
| Episodes | Training episodes per epoch, max epochs | 50, 100 |
| Geometry | Curvature, tangent-norm bound r | -1, 2.0 |
| Variance head | α (Gini weight) | 0.5 |
| Variance head | ν₀ initial value | 0.4 |
| Variance head | Log prior clip, variance floor | [−14, 3], 1e-6 |
| Optimizer | Gradient-norm clip | 5.0 |
| Optimizer | Exponential Moving Average (EMA) decay | min(0.98, (1 + t) / (10 + t)) at step t |
| Prior weights | gφ linear weights (learning rate, weight decay) | 1e-3, 1e-2 |
| Four cross-attention gates | learning rate, weight decay | 1e-2, 0 |


### 5. Results
The proposed model reports test accuracy of 75.17% on PAD-UFES-20 dataset and 78.62% on HAM10000 dataset under 5-shot setting. 
Embeddings of the PAD-UFES-20 test images (image + metadata) from the proposed model is shown in the figure below. (a) The Poincaré ball at true scale; the solid circle is the boundary ‖x‖ = 1, the dashed circle the soft bound tanh 2 ≈ 0.964, and stars are the class Einstein midpoints. (b) Zoom on the occupied region: the radius of each point is exact; its angle is a 2-D principal component projection of log₀(x), so angles are approximate. (c) t-SNE computed on exact pairwise Poincaré distances. Silhouette score on Poincaré distances: 0.28 for the six classes, 0.29 for benign/malignant.
![Figure 2. Visualisation on PAD-UFES-20](result.png)

