using System;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralNetworks
{
	public class NeuralNetworkCompact
	{
		private readonly int[] layers;
		private readonly float[][] neurons;
		private readonly float[][][] weights;

		public NeuralNetworkCompact(int[] layers)
		{
			this.layers = layers;
			neurons = new float[layers.Length][];
			weights = new float[layers.Length][][];

			InitNeurons();
			InitWeights();
		}

		private void InitNeurons()
		{
			for (int layerIndex = 0; layerIndex < layers.Length; layerIndex++)
			{
				var neuronsPerLayer = new List<float>();
				for (int neuronIndex = 0; neuronIndex < layers[layerIndex]; neuronIndex++)
				{
					neuronsPerLayer.Add(0f);
				}

				neurons[layerIndex] = neuronsPerLayer.ToArray();
			}
		}

		private void InitWeights()
		{
			for (int layerIndex = 1; layerIndex < layers.Length; layerIndex++)
			{
				weights[layerIndex] = new float[layers[layerIndex]][];
				for (int neuronIndex = 0; neuronIndex < layers[layerIndex]; neuronIndex++)
				{
					var weightsPerNeuron = new List<float>();
					for (int weightIndex = 0; weightIndex < layers[layerIndex - 1]; weightIndex++)
					{
						weightsPerNeuron.Add(0f);
					}

					weights[layerIndex][neuronIndex] = weightsPerNeuron.ToArray();
				}
			}
		}

		public void Print()
		{
			for (int layerIndex = 0; layerIndex < layers.Length; layerIndex++)
			{
				Debug.Log("LAYER: index: " + layerIndex + " value: " + layers[layerIndex]);
				for (int neuronIndex = 0; neuronIndex < layers[layerIndex]; neuronIndex++)
				{
					Debug.Log("	NEURON: " + neuronIndex + " value: " + neurons[layerIndex][neuronIndex]);
					if (layerIndex == 0) continue;
					for (int weightIndex = 0; weightIndex < layers[layerIndex - 1]; weightIndex++)
						Debug.Log("		WEIGHT: " + weightIndex + " value: " + weights[layerIndex][neuronIndex][weightIndex]);
				}
			}
		}
	}

	public class NeuralNetworkNodebased
	{
		private readonly List<Neuron> allNeurons;
		private readonly Neuron[][] neurons;

		public NeuralNetworkNodebased(int[] layers)
		{
			allNeurons = new List<Neuron>();
			neurons = new Neuron[layers.Length][];
		}

		private struct Neuron
		{
			public float value;
			public float[] outgoingWeights;
			public float[] incomingWeights;
		}
	}
}