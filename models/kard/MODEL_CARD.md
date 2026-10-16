# Deployed MoViNet-A0-Stream + KARD-18

This card describes the checkpoint used by the default dashboard and archived
five-method XAI studies. Later architecture controls and adapters are separate
research models and have not replaced these deployed weights.

| Property | Value |
|---|---|
| Backbone | Official Google MoViNet A0-Stream, pretrained on Kinetics-600 |
| Adaptation | Frozen original backbone, train-only 480D normalization, dropout 0.25 during training, new affine 18-class head |
| Training | 324 KARD clips; subjects 1,3,4,6,7,8 |
| Validation | 108 clips; subjects 5,10; 96 correct, 88.89% accuracy |
| Test | 108 clips; subjects 2,9; 94 correct, **87.04% accuracy**, 86.80% macro F1 |
| Offline input | RGB [0,1], aspect-preserving pad to 172², 5 fps, max 50 frames / centered max 10 s |
| Live window | 20 samples at 5 fps, four seconds, fresh states per window |
| Source head SHA256 | `e2e3bb899d133b1290a891ad0d82e1a6d8205ffb94c2799e6115bb0c99c63a70` |
| Portable head SHA256 | `0e16e03289e5466984922b6f91a89b3ecfd6bc98ee5d01db182f82e0e4017098` |

[Provenance](provenance.json) records original/portable identity, official
backbone hashes and model-specific XAI assets. The portable head is a lossless
serialization change, not retraining. The original Kinetics classifier is
bypassed. The combined ONNX exports are tied to this same historical source.

## Evaluation limits

The split is researcher-defined, not an official KARD benchmark. Each held-out
partition has only two subjects; earlier test results were already observed
during the research sequence. Checkpoint/L2 selection used validation. The
reported clip-level score does not establish continuous-webcam accuracy,
unknown-action rejection, generalization to new datasets, or calibrated
probabilities. There is no idle/unknown class.

XAI studies reused validation after head selection and remain exploratory.
Concept directions are action-family proxies; channel interventions do not
establish complete class-specific circuits. Gradient-based methods require the
TensorFlow model, not just the ONNX logits export.

## Performance and use

Use this model for controlled KARD demonstrations and model-inspection research.
The [deployment benchmark](../../docs/benchmarks/README.md) separates model
invocation and app processing, startup and temporal context. A four-second
input window is required before a complete prediction even if model execution
takes milliseconds. GPU camera behavior remains user-led validation.

The fresh PyTorch control's mean 92.28% test accuracy and the LoRA comparison's
mean 92.90% are different experimental heads, averaged over three seeds. They
are not this checkpoint's scores. See [the study index](../../docs/research/README.md).

The repository provides HAR and XAI tooling, not certified privacy protection.
Data and pretrained-weight terms are described in [third-party notices](../../THIRD_PARTY_NOTICES.md).
