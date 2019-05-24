using System.Collections.Generic;
using System.Text;
using Random = UnityEngine.Random;

namespace NeuralNetworks
{
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