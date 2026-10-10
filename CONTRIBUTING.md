# Contributing

Keep model code inside adapters and XAI code inside methods. Preserve the shared RGB/logit contract. Add a focused test with an independent expected result, plus a config/example when the feature changes integration. Optional runtimes must load lazily so the base installation remains lightweight.

Run `python -m pytest` before proposing a change. For a model/hook change, also compare unchanged replay against the actual original model and record weight/preprocessing hashes. Do not add datasets, raw camera recordings, secrets, local absolute paths or generated caches to Git. Save experiment outputs under the ignored `outputs/` folder.

Report what was actually executed separately from research interpretations. Include failures and control sensitivity. Do not promote synthetic fixtures, subject repetitions or random intervention draws into independent HAR accuracy evidence.
