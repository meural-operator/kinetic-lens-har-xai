# Optional explanations during camera inference

Choose **Technique** in the Explanation Lab. **Explain latest window** makes one request. **Automatically explain new windows** repeats requests after the selected minimum pause and only when a fresh new prediction is available. Automatic mode is off by default. Stop releases the camera, disables automatic mode and clears replay images. There is at most one outstanding explanation per camera session; requests are not queued.

| Technique | Runtime display | What it explains | Main cost or limitation |
|---|---|---|---|
| Spatial Grad-CAM | Animated original sampled window with red heatmap; optional blue rival evidence; overlay toggle and opacity slider | Coarse class-associated evidence at a selected backbone layer | Gradient computation; 6 × 6 maps contain temporal context and are not precise object segmentation |
| Temporal attribution | Intervention table with target logit drops and changed predictions | Effects of reversing, shuffling, freezing or replacing temporal segments | Many inference calls; replacement references affect results |
| Concept attribution | Proxy concept projections, derivatives and deletion effects | Associations and changes along training-fitted feature directions | Bundled action-family proxies are not independently validated semantic concepts |
| Contrastive explanation | Signed feature contributions to the target-versus-rival margin | Exact affine-head decomposition of a class comparison | Feature coordinates are not pixels or named concepts; requires an affine-head hook |
| Mechanistic explanation | Channel removal/retention effects | Effects of candidate internal channel groups under chosen replacements | Multiple backbone replays; bundled rankings cover only Drink, Two hand wave and Forward Kick |

Spatial replay deliberately uses the original samples from the exact explained window. An explanation takes time, so placing its map on the current moving camera frame would misalign the evidence. The live video continues in its own panel. The replay is labeled with window age, sample number and target class. Thumbnails are resized for display and padded in the same square coordinate system as the model input. Replay loops through the available samples at the model sampling rate; it is not a new continuous video recording.

Target defaults to the predicted class. For spatial attribution, leaving the rival unspecified produces positive Grad-CAM; selecting a rival produces a signed class comparison. For contrastive attribution, an unspecified rival uses the runner-up (or actual winner when explaining another target). The heatmap can be hidden to inspect the same sampled images without the overlay.

Explanation work shares the model lock with prediction work. Some explanations are substantially slower than a prediction and can make prediction age increase. Reduce explanation frequency or use one-shot explanations when this matters. Cadence is a minimum pause after completion, not a guaranteed explanation frame rate. Unsupported capabilities are disabled; errors stop automatic mode and display the reason.

## Verification status

The five explanation computations were executed previously against the trained KARD model. Camera capture and prediction were observed in Chrome before these controls were added. At the user's request, the new automatic controls and animated overlay have not undergone camera or runtime testing; they are provided for the user's interactive validation. Their checks in `tests/test_dashboard.mjs` cover scheduling, cleanup and matching-frame rendering without a camera. A heatmap's appearance does not establish explanatory faithfulness or webcam recognition accuracy.
