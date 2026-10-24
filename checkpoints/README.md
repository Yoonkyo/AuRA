# `checkpoints/` — model weights (not shipped)

Download the Grounded-SAM-2 weights used by the segmentation step into
`checkpoints/groundedSAM/` (the default `AURA_GSAM_CHECKPOINTS` in
`config/aura.env`). See [Installation → Grounded-SAM-2 and model
weights](../docs/installation.md#3-grounded-sam-2-and-model-weights) for the download
commands, and [environments/README.md → Model
weights](../environments/README.md#model-weights) for the exact files, URLs,
sizes and sha256 checksums (with a `sha256sum -c` verify block).

The AnyGrasp detection checkpoint and your license do not come with the SDK
clone. You obtain both from the AnyGrasp authors and place them inside your SDK
checkout, not here ([docs/anygrasp_license.md](../docs/anygrasp_license.md),
steps 4 and 5). The checkpoint's size and the sha256 of the copy used for the
paper are listed in the same [Model
weights](../environments/README.md#model-weights) section.
