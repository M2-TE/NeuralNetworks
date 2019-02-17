using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using Random = UnityEngine.Random;

namespace NeuralNetworks
{
	public abstract class NeuralNetwork { }

	[Serializable]
	public struct NeuralNetworkInitSettings
	{
		public int[] layers;

		[Range(0f, 1f)] public float neuronConnectionChance;
		[Range(0f, 1f)] public float weightMutationChance;
		public float weightMaxMutationRange;
	}

	public class NeuralNetworkCompact : NeuralNetwork
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
				neurons[layerIndex] = new float[layers[layerIndex]];
				for (int neuronIndex = 0; neuronIndex < layers[layerIndex]; neuronIndex++)
					neurons[layerIndex][neuronIndex] = 0f;
			}
		}

		private void InitWeights()
		{
			for (int layerIndex = 1; layerIndex < layers.Length; layerIndex++)
			{
				weights[layerIndex] = new float[layers[layerIndex]][];
				for (int neuronIndex = 0; neuronIndex < layers[layerIndex]; neuronIndex++)
				{
					weights[layerIndex][neuronIndex] = new float[layers[layerIndex - 1]];
					for (int weightIndex = 0; weightIndex < layers[layerIndex - 1]; weightIndex++)
						weights[layerIndex][neuronIndex][weightIndex] = 0f;
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


	public class NeuralNetworkNodebased : NeuralNetwork
	{
		private const float minWeight = -1f;
		private const float maxWeight = 1f;

		private readonly List<Neuron> allNeurons;
		private readonly Neuron[][] neurons;
		private readonly int[] layers; // stores amount of neurons per layer

		private readonly float neuronConnectionChance;
		private readonly float weightMutationChance;

		public NeuralNetworkNodebased(NeuralNetworkInitSettings initSettings)
		{
			layers = initSettings.layers;
			neuronConnectionChance = initSettings.neuronConnectionChance;
			weightMutationChance = initSettings.weightMutationChance;

			allNeurons = new List<Neuron>();
			neurons = new Neuron[layers.Length][];

			InitNeurons();
		}

		private void InitNeurons()
		{
			Neuron tempNeuron;
			for (int layerIndex = 0; layerIndex < neurons.Length; layerIndex++)
			{
				neurons[layerIndex] = new Neuron[layers[layerIndex]];
				for(int neuronIndex = 0; neuronIndex < neurons[layerIndex].Length; neuronIndex++)
				{
					var incomingWeights
						= layerIndex != 0
						? new float[layers[layerIndex]]
						: null;

					if (incomingWeights != null)
						for (int weightIndex = 0; weightIndex < incomingWeights.Length; weightIndex++)
							incomingWeights[weightIndex] = Random.Range(0f, 1f) < neuronConnectionChance ? Random.Range(minWeight, maxWeight) : 0f;

					tempNeuron = new Neuron(incomingWeights); // input layer has no incoming weights
					allNeurons.Add(tempNeuron);
					neurons[layerIndex][neuronIndex] = tempNeuron;
				}
			}
		}

		private void ResetValues()
		{
			for (int i = 0; i < allNeurons.Count; i++)
				allNeurons[i].Value = 0f;
		}

		public override string ToString()
		{
			StringBuilder stringBuilder = new StringBuilder();
			for (int layerIndex = 0; layerIndex < neurons.Length; layerIndex++)
			{
				stringBuilder.Append("layer: ");
				stringBuilder.Append(layerIndex);
				for (int neuronIndex = 0; neuronIndex < neurons[layerIndex].Length; neuronIndex++)
				{
					stringBuilder.AppendLine();
					stringBuilder.Append("	");
					stringBuilder.Append(neurons[layerIndex][neuronIndex].ToString());
				}
				stringBuilder.AppendLine();
				stringBuilder.AppendLine();
			}
			return stringBuilder.ToString();
		}

		private class Neuron
		{
			public float Value;
			public float[] IncomingWeights; // weighted connections to nodes of previous layer; null if this is an input neuron

			public Neuron(float[] incomingWeights, float value = 0f)
			{
				Value = value;
				IncomingWeights = incomingWeights;
			}

			public override string ToString()
			{
				StringBuilder stringBuilder = new StringBuilder();
				stringBuilder.Append("value: ");
				stringBuilder.Append(Value);

				if (IncomingWeights != null)
				{
					stringBuilder.Append("	weights:");
					for (int i = 0; i < IncomingWeights.Length; i++)
					{
						stringBuilder.Append("	");
						stringBuilder.Append(i);
						stringBuilder.Append(": ");
						stringBuilder.Append(IncomingWeights[i]);
					}
				}
				return stringBuilder.ToString();
			}
		}
	}
}