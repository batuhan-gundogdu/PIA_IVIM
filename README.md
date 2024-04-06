# Univerity of Chicago Radiology Submission for the AAPM IVIM Challenge

This work uses Physics-Informed Autoencoders for the solution of quantitative IVIM parameters from the simulated breast MR images using the VICTRE breast phantom. https://breastphantom.readthedocs.io/en/latest/. 

The dataset can be found here : https://uchicago.box.com/s/sglrrliwpywe01kwyqoxvgfvvqtxqlmt

![image](https://github.com/batuhan-gundogdu/PIA_IVIM/assets/63497830/26136a6f-73cc-4c04-acc0-1435006cc65e)

Model-1 : Supervised, Using 80% of the training set for supervision, and 20% for validation.
Model-2 : Self-supervised, using samples from training set distribution and simulated samples.
Model-3 : Unsupervised, using samples from training set distribution.
Model-4 : Unsupervised, using samples from physically-feasible distribution
Model-5 : Variational, estimating the domain distribution
Model-6 : Mixture of Experts
