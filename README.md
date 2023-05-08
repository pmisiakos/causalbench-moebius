# CausalBench ICLR-23 Challenge

We provide the implementation of MÖBIUS method submitted to the ICLR 2023 CausalBench competition. Our work with title *"Learning Gene Regulatory Networks under Few Root Causes Assumption"*  was awarded the 3rd prize in the competition and was presented in the Machine Learning for Drug Discovery workshop.

## Install

```bash
pip install -r requirements.txt
```

## Use

### How to Evaluate MÖBIUS?

To run a custom graph inference function, set `--model_name="custom"` and `--inference_function_file_path` to the file path that contains your custom graph inference function (e.g. [grnboost.py](../causalbench-starter-main%205/src/grnboost.py) in this repo). You are given two starter implementations to choose from in src/, grnboost.py and dcdi.py. Your mission is to choose one of them and fine tune them to improve their performance. Hints on potential ways to improve the methods can be found directly in the code. 

You should evaluate your method with the following command:

```bash
causalbench_run \
    --dataset_name weissmann_rpe1 \
    --output_directory ./output \
    --data_directory /path/to/data/storage \
    --training_regime partial_interventional \
    --partial_intervention_seed 0 \
    --fraction_partial_intervention $FRACTION \
    --model_name custom \
    --inference_function_file_path ./src/main.py \
    --subset_data 1.0 \
    --model_seed 0 \
    --do_filter
```

<!-- ## Citation

Please consider citing, if you reference or use our methodology, code or results in your work: -->


### License

[License](LICENSE.txt)

### Authors

Panagiotis Misiakos,
Chris Wendler and
Markus Püschel

Computer Science Department,
ETH Zurich.
