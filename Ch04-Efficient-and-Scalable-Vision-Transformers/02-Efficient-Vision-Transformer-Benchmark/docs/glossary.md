# Glossary

**Attention window**
: A bounded spatial group of tokens that attend to one another. Swin uses windows to avoid global
  attention over the entire image at every block.

**Automatic mixed precision (AMP)**
: A CUDA execution mode that uses lower precision where safe and gradient scaling during training,
  reducing memory and often increasing speed.

**Backbone**
: The feature-producing portion of a model before its task-specific classifier head.

**Checkpoint**
: Saved model state. This project also stores optimizer, schedule, scaler, random state, and
  configuration identity so training can resume.

**Data efficiency**
: Achieving useful generalization with less training data or supervision. It is not the same as low
  inference latency.

**Distillation token**
: DeiT's additional learned token whose classifier can learn from teacher targets. At inference its
  logits are averaged with the class-token head in this project.

**Fine-tuning**
: Adapting pretrained weights to a new task. Here, all backbone parameters and new 100-class heads
  are trainable, with different learning rates.

**FLOPs / MACs**
: Static arithmetic estimates. Conventions differ: this project labels fvcore's multiply-add-as-one
  convention as MAC-style operations and records unsupported operators.

**Hard distillation**
: Training from the teacher's argmax class rather than its full probability distribution. The
  teacher target is a class ID used in cross-entropy.

**Hierarchy**
: A sequence of feature stages whose spatial resolution decreases while channel capacity increases.
  Swin creates this with patch merging.

**Latency**
: Time for one batch forward. Batch-1 latency is especially relevant to interactive inference.

**Manifest**
: The persisted train/validation/test indices and checksum that make the data split auditable.

**Patch embedding**
: Conversion of image regions into token vectors, often implemented as a strided convolution.

**Patch merging**
: Swin's downsampling operation that combines neighboring tokens and increases channel capacity.

**Peak allocator memory**
: Maximum tensor memory reported by the framework allocator during a measured section. It excludes
  some driver, runtime, and whole-process memory.

**Preset fingerprint**
: SHA-256 hash of the fully resolved experiment configuration. It isolates artifacts and prevents
  incompatible checkpoint resumption.

**Seed**
: Initial value for pseudo-random generators. Matching seeds reduces avoidable variation, but does
  not guarantee bit-identical results across hardware and software stacks.

**Shifted window**
: Swin's alternating window partition. A shift allows tokens separated by one layer's fixed window
  boundary to exchange information in the next.

**Teacher / student**
: A teacher produces auxiliary targets; a student learns from them. The frozen ConvNeXt teacher is
  larger than the Distilled DeiT student's compact classification heads, though model sizes are not
  required to follow one universal rule.

**Throughput**
: Images processed per second for a stated batch size and precision. It is not the reciprocal of a
  universally meaningful per-image latency when batching is involved.

**Top-1 / top-5 accuracy**
: Percentage of examples whose true class is the highest-scoring prediction or appears among the
  five highest-scoring predictions.

**Transfer learning**
: Reusing representations learned on an upstream dataset for a downstream task such as CIFAR-100.
