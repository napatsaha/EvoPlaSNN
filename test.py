from snn import SNN, RandomSpikeGenerator, SNNSimulator, PatternSpikeGenerator
from lrule import STDP_Rule


if __name__ == "__main__":
    # Simulation parameters
    T = 200
    dt = 1e-3

    # Generator parameters
    ascending = True
    interval = 1
    spacing = 10

    # Network parameters
    input_size = 10
    hidden_size = [7]
    output_size = 5

    # Neuron parameters
    tau_mem = 3 * dt
    threshold = 1.0 
    membrane_start = 0.0
    reset_mechanism = "zero"
    tau_trace = 3 * dt # Time constant for the trace
    trace_amp = 0.5
    trace_type = "dx3"

    # STDP parameters
    mu = 0.0 # Power constant for F function
    lambd = 1.0 # Learning rate
    alpha = 1.0 # Asymmetry factor (Higher values favour LTD)

    neuron_params = dict(
        tau_mem=tau_mem,
        threshold=threshold,
        membrane_start=membrane_start,
        reset_mechanism=reset_mechanism,
        tau_trace=tau_trace,
        trace_amp=trace_amp,
        trace_type=trace_type,
    )

    # Set up objects
    lrule = STDP_Rule(mu, lambd, alpha, dt)
    network = SNN(input_size, hidden_size, output_size, dt=dt, learning_rule=lrule, neuron_params=neuron_params)
    spikegen = PatternSpikeGenerator(input_size, ascending=ascending, interval=interval, spacing=spacing)
    simulator = SNNSimulator(network, num_steps=T, spike_generator=spikegen, record_weights=True)

    simulator.run()

    # Plot membrane potentials
    # simulator.mem_recorder.plot(threshold, figtitle="Membrane Potentials", show=True)
    simulator.plot_membranes(plot_inputs=False, cmap="GnBu", cmap_range=(0.5, 1.0))
    simulator.plot_membranes(plot_inputs=True, color="green")

    simulator.plot_traces(cmap="viridis", x_eps=0.0, y_eps=0.1, y_scale=1.0, cmap_range=(0.0, 0.8), trace_scale=0.7)

    simulator.plot_spikes(y_scale=0.5)

    simulator.plot_weights()