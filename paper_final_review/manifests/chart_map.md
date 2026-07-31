# Chart map

| Figure | Question | Family | Fields | Supported claim |
|---|---|---|---|---|
| rq1_unseen_success.svg | Does Full AASSR generalize zero-shot? | Comparison/ranking bar | dependency length, condition, success | Full AASSR frozen unseen success is zero |
| rq1_learning_auc.svg | Does learning occur during training? | Comparison/ranking bar | dependency length, condition, AUC | Full learns but trails contextual/Prophecy |
| transfer_adaptation_curves.svg | Is adaptation faster after transfer? | Ordered line | budget, condition, success | Curves differ only slightly |
| ablation_axis_learning_auc.svg | Which imagination settings differ? | Comparison/ranking bar | axis level, marginal AUC | L6-only matrix effects |
| creativity_funnel.svg | How many strategies pass each creativity gate? | Progression bar | gate, count | Novelty gate removes all candidates |
| storage_by_experiment.svg | Why are artifacts large? | Comparison/ranking bar | experiment, GiB | Ablation and autonomy dominate storage |

Palette policy: single blue root with neutral axes and direct labels. All absolute-magnitude bars start at zero. SVGs were selected because the user requested a lightweight local review package and no plotting runtime is installed.
