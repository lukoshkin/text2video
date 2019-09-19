# References to Some of the Ideas Used in This Work

**0-GP**:   [Improving Generalization and Stability of Generative Adversarial Networks](https://arxiv.org/pdf/1902.03984.pdf)  
        - Gradient exploding in the discriminator\* can lead to mode collapse in the generator (math. justification in the article)  
        - The number of modes in the distribution grows linearly with the size of the discriminator -> higher capacity discriminators are needed for
        better approximation of the target distribution.  
        - Generalization is guaranted if the discriminator set is small enough.  
        - To smooth out the loss surface one can build a discriminator that makes the judgement on a mixed batch of fake and real samples, determining
        the proportion between them (Lucas et al., 2018)  
        - VEEGAN (Srivastava et al., 2017) uses the inverse mapping of the generator to map the data to the prior distribution. The mismatch between
        the inverse mapping and the prior is used to detect mode collapse. It is not able to help, if the generator can remember the entire dataset  
        - Generalization capability of the discriminator can be estimated by measuring the difference between its performance on the training dataset
        and a held-out dataset  
        - When generator starts to produce samples of the same quality as the real ones, we come to the situation where the discriminator has to deal
        with mislabeled data: generated samples, regardless of how good they are, are still labeled as bad ones, so the discriminator trained on such
        dataset will overfit and not be able to teach the generator  
        - Heuristically, overfitting can be alleviated by limiting the number of discriminator updates per generator update. Goodfellow et al. (2014)
        recommended to update the discriminator once every generator update  
        - It is observed that the norm of the gradient w.r.t. the discriminator’s parameters decreases as fakes samples approach real samples. If the
        discriminator’s learning rate is fixed, then the number of gradient descent steps that the discriminator has to take to reach eps-optimal
        state should increase. *Alternating gradient descent with the same learning rate for discriminator and generator, and fixed number of
        discriminator updates per generator update (Fixed-Alt-GD) cannot maintain the (empirical) optimality of the discriminator*. In GANs trained
        with Two Timescale Update Rule (TTUR) (Heusel et al., 2017), the ratio between the learning rate of the discriminator and that of the
        generator goes to infinity as the iteration number goes to infinity. Therefore, the discriminator can learn much faster than the generator
        and might be able to maintain its optimality throughout the learning process.  
        ____________________________________  
        **\*** in case of emperically optimal D  
**CBN**:  
    1. [Modulating early visual processing by language](https://arxiv.org/pdf/1707.00683.pdf)  
    1. [A Learned Representation For Artistic Style](https://arxiv.org/pdf/1610.07629.pdf)  
**BN**:     [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)  
**ResBlocks**:  [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)  
**ProjDisc**:   [cGANs with Projection Discriminator](https://arxiv.org/pdf/1802.05637.pdf)  
**ConvGRU**:    [Convolutional Gated Recurrent Networks for Video Segmentation](https://arxiv.org/pdf/1611.05435.pdf)  
**Basic Ideas for Text Encoders**:  [Realistic Image Generation using Region-phrase Attention](https://arxiv.org/pdf/1902.05395.pdf)  
**D/G Blocks' Structure**:     [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/pdf/1809.11096.pdf)  
        - Employing Spectral Normalization in G improves stability, allowing for fewer D steps per iteration.  
        - Greater batch size can help dealing with mode collapse and impove the network performance, though it might lead to training collapse (NaNs)  
**Joint Structured Embeddings**:  
    1. [Learning Deep Representations of Fine-Grained Visual Descriptions](https://arxiv.org/pdf/1605.05395.pdf)  
    1. [also](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Akata_Evaluation_of_Output_2015_CVPR_paper.pdf)  
**Concatenate by Stacking**:    [StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/pdf/1710.10916.pdf)  
**Self-Attention**:     [A Structured Self-attentive Sentence Embedding](https://arxiv.org/pdf/1703.03130.pdf)  
