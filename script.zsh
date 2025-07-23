run_name="25-07-11_weight-and-bias"
mkdir -p results/binary_es/$run_name
cp script.zsh results/binary_es/$run_name/script.zsh

bias=(true false)
use_weights=(true false)
reward_type=("weighted" "mapped")

# Loop through the combinations of bias and use_weights
for b in "${bias[@]}"; do
  for w in "${use_weights[@]}"; do
    for reward in "${reward_type[@]}"
    do
      echo "Running with bias=$b and use_weights=$w"
      python src/run_evo_lrule.py --parent $run_name --config "binary_es_v2_$reward.yaml" --override arule_params.use_weights=$w arule_params.bias=$b arule_params.learning_rate=0.1
    done
  done
done

# Run evaluations
python src/eval_group.py $run_name --num_trials 100