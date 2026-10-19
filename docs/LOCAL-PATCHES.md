# Local deployment patches

This fork tracks `0rand/main` and adds the following changes. Private `.env`,
API keys, host addresses, deployment automation and runtime state stay local.

- **Image limit:** the display-KV build forwards the existing `--vision-max-images`
  option to DeepSeek's `DsVision`. `VISION_MAX_IMAGES` defaults to 8; this deployment
  uses 32. Requests above the configured count still fail validation. More images
  consume more prompt tokens and memory; this setting does not increase context.
- **Concurrent image crash:** completed prefill streams are removed by object
  identity. Dataclass equality compared image tensors during `list.remove`, causing
  `Boolean value of Tensor with more than one value is ambiguous` and killing the
  lane. The regression reproduces the original crash with CPU tensors and tests
  all completion positions for one through four concurrent image requests.
- **DRM preparation:** an idle device already using `modeset=1 fbdev=0` is reused
  without unloading its module. Compute, mapping, FD-owner and scanout checks
  still run, parameters are rechecked, and paths requiring an unload retain the
  module-refcount guard. Display-manager handling stays unchanged.
- **Build and launch settings:** `DISPLAY_BASE_IMAGE` selects a pristine base
  independently of the active image, `DISPLAY_PREFILL_CHUNK` optionally overrides
  the bundle's 512 default, and `SERVED_ALIAS_2` supports an additional router ID.

Both engine fixes run after upstream's pinned, fingerprint-checked transformer.
They validate exact anchors and Python syntax before writing either source file.
The image fingerprint includes the fixes, so rebuilds cannot silently reuse an
image built without them. A second install or source drift fails the build.

## Verification

```bash
python3 tests/display_drm_test.py
python3 tests/start_display_kv_test.py
TF_DISPLAY_SOURCE=/path/to/pristine/TensorFold python3 -m unittest discover -s experiments/display-kv/tests -v
```

The engine tests need the pinned TensorFold source and its Python dependencies
(CPU PyTorch is sufficient). They load no model weights. After rebuilding and
starting both nodes, run `./verify.sh` and concurrent image requests against the
live server. CPU regressions alone do not establish GPU serving stability.

Verified on 2026-10-19 against upstream `172cf78017f7781e234612ef2cbe8b75190616dc`:
20 DRM tests passed, 20 launcher tests passed, and 96 engine tests completed
successfully (four CUDA-only checks skipped in the CPU run). The Dockerfile also
built successfully from `tensorfold-dsv41:07aa02585f64`, applying upstream's
transformer followed by both local engine fixes. Live GPU validation of this
combined build remains separate.
