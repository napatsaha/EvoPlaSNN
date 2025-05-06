from snn import SNN, RandomSpikeGenerator, SNNSimulator, PatternSpikeGenerator


if __name__ == "__main__":
    # Parameters
    input_size = 4
    hidden_size = [5, 5]
    output_size = 2
    tau_mem = 1e-3
    tau_trace = 1e-3
    dt = 1e-3
    threshold = [1.0, 0.5, 0.5, 0.5]

    total_timesteps = 100

    # Create a spike generator
    # spike_gen = RandomSpikeGenerator(input_size, dist="binomial", p=0.02)
    spike_gen = PatternSpikeGenerator(input_size, interval=5)

    # Create a spiking network
    network = SNN(input_size, hidden_size, output_size, dt=dt, tau_mem=tau_mem, tau_trace=tau_trace, threshold=threshold, reset_mechanism="zero", include_input_layer=True)

    # Create a simulator
    simulator = SNNSimulator(network, total_timesteps, spike_gen)
    simulator.run()

    # Plot membrane potentials
    simulator.mem_recorder.plot(threshold, figtitle="Membrane Potentials", show=True)