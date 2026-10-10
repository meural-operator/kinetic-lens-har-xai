# Independent semantic validation: proposed next study

This is a proposed annotation and evaluation protocol, not an experiment already performed. Current action-family proxy results must remain labeled partial semantic validation.

## What to annotate

Annotate observable attributes from the video rather than copy the action name. Candidate attributes include arm elevation, unilateral versus bilateral arm movement, visible leg extension, visible hand-object contact, object presence/type, trunk inclination, and sitting-to-standing transition. Define each attribute operationally with positive, negative and uncertain examples before annotation. Some attributes need a temporal interval rather than one frame. Attribute definitions and threshold choices are domain decisions; this file does not assert that a particular geometric threshold defines an action.

Record clip identity, sampled frame/timestamp interval, attribute value, visibility/occlusion, annotator identifier, uncertainty, and optional person/body-part masks. Annotators should not see model predictions, attribution maps or proxy scores while labeling. An ambiguous/occluded attribute should be unknown rather than forced negative. The existing class name may remain in the dataset metadata, but should not automatically determine concept labels.

## Construct controls

Include the same attribute in multiple actions, and its absence within actions when natural examples exist. Balance or match subject, camera, clothing/background and object context as far as the dataset permits. Keep human labels and matching decisions independent of the model's heatmaps. Some KARD attributes may remain confounded with action identity; document this instead of inventing within-action variation.

Double-annotate a subset and report agreement using a measure appropriate to binary, categorical or continuous labels. Retain disagreements and visibility failures. Use independent subjects/data for final assessment; the current validation split has already been reused for head selection and XAI follow-up.

## Evaluate claims separately

1. **Availability:** can a probe detect the independent attribute on held-out subjects and unseen action identities?
2. **Use:** does the model's class score depend on a direction or subspace carrying that attribute, beyond matched random/association controls?
3. **Localization:** do maps align with independently annotated actor/body-part/object regions, and do carefully matched input interventions have consistent effects?
4. **Specificity:** does the effect survive controls that alter correlated attributes or background while preserving the proposed concept?
5. **Mechanism:** do selected channels/paths carry and mediate the effect across inputs, references and layers, with removal, retention and restoration controls?

A high probe AUC alone answers availability under a chosen labeling definition. It does not establish use, specificity, localization or a complete circuit. A physically meaningful edited video or paired acquisition experiment can provide stronger semantic evidence than an arbitrary representation-space deletion, but the edit's realism and confounds must also be evaluated.
