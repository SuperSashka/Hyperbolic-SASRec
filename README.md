# Instructions to run experiments

## Preparing datasets

 To initiate datasets preparation process, one needs to call the corresponding script with the following command:
```shell
python scripts/get_data.py --dataset=<dataset name> [--pcore]
```

Supported datasets:
- Movielens (https://grouplens.org/datasets/movielens):
  - ml-1m, ml-10m, ml-20m, ml-25m, ml-latest
- Amazon (https://nijianmo.github.io/amazon/index.html):
  - Amazon_Fashion, All_Beauty, Appliances, Arts_Crafts_and_Sewing, Automotive, Books, CDs_and_Vinyl, Cell_Phones_and_Accessories, Clothing_Shoes_and_Jewelry, Digital_Music, Electronics, Gift_Cards, Grocery_and_Gourmet_Food, Home_and_Kitchen, Industrial_and_Scientific, Kindle_Store, Luxury_Beauty, Magazine_Subscriptions, Movies_and_TV, Musical_Instruments, Office_Products, Patio_Lawn_and_Garden, Pet_Supplies, Prime_Pantry, Software, Sports_and_Outdoors, Tools_and_Home_Improvement, Toys_and_Games, Video_Games

 An example of the corresponding command for the `Movielens-1M` dataset:
```shell
python scripts/get_data.py --dataset=ml-1m
```

## Tuning hyper-parameters

To start hyper-parameter tuning, use the command python `tune.py` with the following input arguments, either in the format of
`--parameter_name=parameter_value` or simply `--parameter_name` for boolean parameters.

The list of accepted input arguments is the following:
- `model` The name of the model.
- `dataset` The name of the dataset. Note that when the --pcore setting is provided (default for Amazon datasets as they are 5-core in the source storage), the name of the dataset must be appended with the pcore value (i.e., 5 by default). For example, the Amazon `Books` datasets must be referenced as `Books_5`, if a different pcore value was not explicitly provided.
- `time_offset` A real number between zero and one. Used for splitting sequential observations into train and test. The time interval of observations is split in proportion to the time offset, which is simply a quantile of observations up to the splitting timepoint. 
- `config_path` The path to the config file. Sample configs for various models are provided in the ```grid``` folder. For example, to use the default config for `hypsasrec`, use the option `--config_path="./grids/hypsasrec.py"`.
- `grid_steps` A positive integer number. Determines the total number of configurations to be tried during the grid search.
- `check_best` Include this parameter, without any parameter values, to see the final results on the test data for the best configuration found.

Note that the `model` naming in the code is different form the one used in the paper:
- `sasrec` is the BCE-based Euclidean model
- `sasrecb` is the CE-based Euclidean model
- `hypsasrecb` is the BCE-based hyperbolic model
- `hypsasrec` is the CE-based hyperbolic model

An example of a full command:
```shell
python tune.py --model=sasrec --dataset=ml-1m --time_offset=0.95  --config_path="./grids/sasrec.py" --grid_steps=60
```

## Reproducibility
Please see the `conda.yml` file for the vesrions of the Python packages used in the experiments. The best configs for several datasets are provided in ```best``` folder.
