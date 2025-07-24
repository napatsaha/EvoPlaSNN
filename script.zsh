run_name="25-07-24_binary_jitter_failure"
run_dir="results/exp01/$run_name"
mkdir -p $run_dir
cp script.zsh $run_dir/script.zsh

jitter=(0.0 0.5 1.0)
failure=(0.0 0.05 0.1)

# Loop through the combinations of bias and use_weights
for jtd in "${jitter[@]}"; do
  for fal in "${failure[@]}";
    do
      # echo "Running with bias=$b and use_weights=$w"
      echo "Running with jitter_std=$jtd and failure_rate=$fal"
      python src/run_evo_lrule.py --parent $run_dir --config "config_binary.yaml" --override spikegen_params.jitter_std=$jtd spikegen_params.failure_rate=$fal 
  done
done

# Run evaluations
python src/eval_group.py $run_dir --num_trials 10