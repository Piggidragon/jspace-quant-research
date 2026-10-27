"""Robustness of Jacobian-lens (J-space) readouts to weight quantization.

Pipeline (see docs/en/02-methodology.md):

    capture  — forward passes per model over all prompts; store the residual stream of every
               layer at the readout positions (GPU, jlens-server)
    analyze  — compute the readouts of every experimental arm (model × lens × unembedding) and
               derive metrics (CPU, numpy)
    plot     — figures, in English and German
"""
