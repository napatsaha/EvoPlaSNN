run_name="25-07-10_weight-and-bias"
mkdir results/binary_es/$run_name
cp script.zsh results/binary_es/$run_name/script.zsh

bias=(true false)
use_weights=(true false)
reward_type=(weighted mapped)

# Loop through the combinations of bias and use_weights
for b in "${bias[@]}"; do
  for w in "${use_weights[@]}"; do
    echo "Running with bias=$b and use_weights=$w"
    python src/run_evo_lrule.py --parent $run_name --config binary_es_v2_$reward_type.yaml --override arule_params.use_weights=$w arule_params.bias=$b arule_params.learning_rate=0.1
  done
done
# python src/run_evo_lrule.py --parent $run_name --config binary_es_v2_weighted.yaml --override arule_params.use_weights=True arule_params.bias=True