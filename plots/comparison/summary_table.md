# Cross-Model Summary

| model                  |   num_layers |   best_test_pearson |   best_test_spearman |   best_layer |   probe_gradient_probe_delta |   probe_gradient_bm_delta |   random_probe_delta |   random_bm_delta |   logit_gradient_probe_delta |   logit_gradient_bm_delta |   num_prompts |
|:-----------------------|-------------:|--------------------:|---------------------:|-------------:|-----------------------------:|--------------------------:|---------------------:|------------------:|-----------------------------:|--------------------------:|--------------:|
| gpt2-small             |           12 |              0.5185 |               0.5193 |            8 |                      -0.2444 |                   -0.0418 |              -0.0489 |           -0.0762 |                       0      |                   -0.0372 |           299 |
| EleutherAI/pythia-410m |           24 |              0.6109 |               0.5603 |           16 |                       0.1407 |                    0.0053 |               0      |            0.0079 |                       0.1407 |                    0.0053 |           299 |
