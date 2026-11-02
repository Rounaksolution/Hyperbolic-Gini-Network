## Contents
1. [The Problem Statement](#the-problem)
2. [Methodology](#methodology)
3. [Dataset Preprocessing](#dataset)
4. [Comparison with existing Work](#comparison)
5. [Conclusion](#conclusion)


## 1. The Problem Statement
A nearest-prototype rule is wrong whenever classes differ in spread, and with limited support data, the spread is the thing we know least about.

**Why spread matters.** Skin lesion classes are not equally varied. For example, NEV in PAD-UFES-20 is a fairly uniform class, while SEK and ACK cover flat, raised, pigmented and keratotic presentations. A query sitting at the edge of a broad class can be closer to the centre of a tight neighbour.


## 2. Methodology

**Why Poincaré ball**
The labels form a tree: root, then benign and malignant, then six/seven diagnoses (and subtypes below those). In hyperbolic space the volume of a ball grows exponentially with its radius, as the number of nodes in a tree grows with depth. Trees therefore embed with very low distortion in a few hyperbolic dimensions, while Euclidean space needs many more dimensions for the same distortion. Superclasses can sit nearer the origin and fine classes towards the boundary. 

**Prototype charts.**
Why is variance measured after a Möbius shift? In raw ball coordinates the same coordinate step means a larger true distance near the boundary than near the centre. A variance computed there would mix real spread with position in the ball. We first move the prototype to the origin with the isometry h ↦ (−p) ⊕ h, then take the log map. In these coordinates 2‖ξ‖ equals the true hyperbolic distance. The variance is estimated and used in the same chart, so it means the same thing in both places. 

**The metadata-conditioned variance prior**
The prior answers a question the embedding cannot: given who these support patients are, how widely should lesions like theirs be expected to vary? 

Figure 1 illustrates the proposed Metadata-Conditioned Shrinkage Priors for Hyperbolic Prototypical Few-Shot Skin Lesion Classification model. 

## 3. Dataset
Two publicly available skin lesion dataset namely PAD-UFES-20 and HAM10000 have been used. Each image is resized to 224×224 without cropping and encoded under six dihedral views: the identity, horizontal and vertical flips, and rotations by 90°, 180° and 270°. Because both backbones CLIP and DINOv2 are frozen, the features of all six views are computed once and cached. During training, each sampled image uses one view drawn uniformly at random; evaluation uses the identity view only.
For hierarchical structure of PAD-UFES-20 dataset, **benign**: ['ACK', 'NEV', 'SEK'], **malignant**: ['BCC', 'MEL', 'SCC'] is considered. ACK is pre-malignant and is considered benign class. For HAM10000 dataset, **benign**: ['bkl', 'df', 'nv', 'vasc'], **malignant**: ['akiec', 'bcc', 'mel'] is considered.

## 4. Comparison with existing Work
The proposed model reports test accuracy of 75.17% on PAD-UFES-20 dataset and 78.62% on HAM10000 dataset under 5-shot setting. 
