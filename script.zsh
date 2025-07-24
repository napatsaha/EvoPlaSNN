run_name="25-07-24_binary_jitter_failure"
mkdir -p results/exp01/$run_name
cp script.zsh results/exp01/$run_name/script.zsh

jitter=(0.0 0.5 1.0)
failure=(0.0 0.05 0.1)

# Loop through the combinations of bias and use_weights
for jtd in "${jitter[@]}"; do
  for fal in "${failure[@]}";
    do
      # echo "Running with bias=$b and use_weights=$w"
      python src/run_evo_lrule.py --parent $run_name --config "config_binary.yaml" --override spikegen_params.jitter_std=$jtd spikegen_params.failure_rate=$fal 
  done
done

# Run evaluations
python src/eval_group.py $run_name --num_trials 100