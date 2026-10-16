"""Versioned protocol for the first KARD 25%-training-split experiment."""
from pathlib import Path
import os

ROOT=Path(__file__).resolve().parent
DATASET=ROOT.parent/'datasets'/'KARD'
FRACTION=float(os.environ.get('KARD_TRAIN_FRACTION','0.25'))
if FRACTION not in (0.25,0.5,1.0):
    raise ValueError('Supported KARD_TRAIN_FRACTION values: 0.25, 0.5 or 1.0')
TARGET_FPS=float(os.environ.get('KARD_TARGET_FPS','5'))
if TARGET_FPS not in (5.0,10.0) or (TARGET_FPS==10.0 and FRACTION!=1.0):
    raise ValueError('Supported sampling: 5 fps for all fractions, or 10 fps with the full training pool')
OUT=ROOT/(f'kard_a0_head_{round(FRACTION*100)}pct'+('_10fps' if TARGET_FPS==10.0 else ''))
OUT.mkdir(exist_ok=True)
SEED=42
LABELS=['Horizontal arm wave','High arm wave','Two hand wave','Catch Cap',
        'High throw','Draw X','Draw Tick','Toss Paper','Forward Kick','Side Kick',
        'Take Umbrella','Bend','Hand Clap','Walk','Phone Call','Drink','Sit down','Stand up']
PROTOCOL={'version':1,'seed':SEED,'model':'a0_stream','resolution':172,
    'target_fps':5.0,'max_frames':50,'stream_chunk_frames':8,
    'preprocessing':'RGB float32 [0,1]; bilinear resize with padding to 172x172; entire clip at 5 fps if <=10 sec, centered contiguous 10-second window otherwise.',
    'train_fraction_of_training_pool':0.25,'subject_counts':{'train':6,'validation':2,'test':2},
    'selection':'Seeded subject partition; exactly 81/324 training clips; 4 or 5 per class, at most one repetition per subject/class; greedy balance subjects with seeded tie breaking, followed by seeded assignment of exactly 27 clips per repetition. File integrity checked before selection; no prediction/feature-based selection.',
    'head':'Frozen official 480-dimensional pooled A0 backbone endpoint; train-only feature standardization; dropout(0.25); Dense(18) logits with L2 regularization. No training of backbone or original Kinetics classifier.',
    'training':'Adam(learning_rate=0.001), batch size 16, maximum 200 epochs, patience 25; three prespecified L2 strengths [0.0001,0.001,0.01]; checkpoint and candidate selection by validation macro F1, then validation cross entropy. Test evaluated once for selected candidate.',
    'augmentation':'No video augmentation in this first frozen-feature baseline. Head dropout during training only.',
    'limitations':'One researcher-defined subject split, not a published KARD benchmark protocol; small labelled subset; no automated semantic verification of labels, no backbone fine-tuning, no cross-validation.'}
if FRACTION==0.5:
    PROTOCOL={**PROTOCOL,'version':2,'train_fraction_of_training_pool':0.5,
        'selection':'Nested extension of the original 81 training clips to 162/324. Integer min-cost flow balances exactly 9 per class, 27 per training subject, 54 per repetition, and 1 or 2 clips per class/subject pair. Seeded costs; no prediction-based selection.',
        'previous_test_results_already_observed':True,
        'comparison':'Prespecified next experiment increases training fraction only. Same split, input policy, head, seed, optimizer and validation-only model-selection rule. Retrain head from initialization; recompute normalization on the larger training subset.',
        'limitations':'One fixed researcher-defined subject split; test results from the 25% baseline were already observed. This is a planned comparison on the same test set, not a newly blind test. No backbone fine-tuning or cross-validation.'}
if FRACTION==1.0:
    PROTOCOL={**PROTOCOL,'version':3,'train_fraction_of_training_pool':1.0,
        'selection':'All 324 clips from the six fixed training subjects: 18 per class, 54 per subject and 108 per repetition. Includes both earlier nested subsets. Validation and test subjects unchanged.',
        'previous_test_results_already_observed':True,
        'comparison':'Prespecified increase to the full training split only. Same input policy, frozen backbone, head, seed, optimizer and validation-only selection. Head initialized afresh; normalization fitted on 324 training clips only.',
        'limitations':'One researcher-defined subject split; earlier test results already observed. This is a planned comparison on the same test set, not a newly blind test. No backbone fine-tuning or cross-validation.'}
if TARGET_FPS==10.0:
    PROTOCOL={**PROTOCOL,'version':4,'target_fps':10.0,'max_frames':100,
        'preprocessing':'RGB float32 [0,1]; bilinear resize with padding to 172x172; entire clip at 10 fps if <=10 sec, centered contiguous 10-second window otherwise.',
        'controlled_sampling_comparison':True,'baseline_experiment':'kard_a0_head_100pct',
        'comparison':'Change target sampling from 5 to 10 fps and cap from 50 to 100 frames, retaining the same centered 10-second window, clips, subject split, frozen official backbone, head architecture, seed, optimizer and validation-only selection. Head initialized afresh; standardization refitted on the new training features.',
        'selection_across_sampling_policies':'Recommend the sampling policy by validation macro F1, then lower validation cross entropy, before creating 10-fps test features. Test results remain descriptive on a previously observed split.'}
