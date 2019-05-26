using System;
using System.Text;
using UnityEngine;
using Random = UnityEngine.Random;

namespace NeuralNetworks
{
	public class NeuralNetworkCompact : NeuralNetwork
	{
		private readonly int[] layerSizes;
		private readonly float[][] neurons;
		private readonly float[][][] weights;

		public NeuralNetworkCompact(int[] layers, float mutationChance, float maxMutation)
		{
			this.mutationChance = mutationChance;
			this.maxFlatMutation = maxMutation;

			layerSizes = layers;
			neurons = new float[layers.Length][];
			weights = new float[layers.Length][][];

			InitNeurons();
			InitWeights();
		}

		public NeuralNetworkCompact(NeuralNetworkCompact deepCopyBlueprint)
		{
			mutationChance = deepCopyBlueprint.mutationChance;
			maxFlatMutation = deepCopyBlueprint.maxFlatMutation;

			layerSizes = new int[deepCopyBlueprint.layerSizes.Length];
			CopyLayerSizes(deepCopyBlueprint);

			neurons = new float[layerSizes.Length][];
			weights = new float[layerSizes.Length][][];

			InitNeurons();
			CopyWeights(deepCopyBlueprint);
		}

		private void CopyLayerSizes(NeuralNetworkCompact deepCopyBlueprint)
		{
			for (int i = 0; i < layerSizes.Length; i++)
			{
				layerSizes[i] = deepCopyBlueprint.layerSizes[i];
			}
		}

		private void CopyWeights(NeuralNetworkCompact deepCopyBlueprint)
		{
			for (int layerIndex = 1; layerIndex < layerSizes.Length; layerIndex++)
			{
				weights[layerIndex] = new float[layerSizes[layerIndex]][];
				for (int neuronIndex = 0; neuronIndex < layerSizes[layerIndex]; neuronIndex++)
				{
					weights[layerIndex][neuronIndex] = new float[layerSizes[layerIndex - 1]];
					for (int weightIndex = 0; weightIndex < layerSizes[layerIndex - 1]; weightIndex++)
					{
						weights[layerIndex][neuronIndex][weightIndex] = deepCopyBlueprint.weights[layerIndex][neuronIndex][weightIndex];
					}
				}
			}
		}

		private void InitNeurons()
		{
			for (int layerIndex = 0; layerIndex < layerSizes.Length; layerIndex++)
			{
				neurons[layerIndex] = new float[layerSizes[layerIndex]];
			}
		}

		private void InitWeights()
		{
			// neurons have weights to their previous layer, thus start at layer 1
			for (int layerIndex = 1; layerIndex < layerSizes.Length; layerIndex++)
			{
				weights[layerIndex] = new float[layerSizes[layerIndex]][];
				for (int neuronIndex = 0; neuronIndex < layerSizes[layerIndex]; neuronIndex++)
				{
					weights[layerIndex][neuronIndex] = new float[layerSizes[layerIndex - 1]];
					for (int weightIndex = 0; weightIndex < layerSizes[layerIndex - 1]; weightIndex++)
					{
						weights[layerIndex][neuronIndex][weightIndex] = 0f;
					}
				}
			}
		}

		public override float[] RequestDecision(float[] inputs)
		{
			// set input neurons
			for (int inputIndex = 0; inputIndex < inputs.Length; inputIndex++)
			{
				neurons[0][inputIndex] = inputs[inputIndex];
			}

			//iterate over all neurons and compute feedforward values
			for (int layerIndex = 1; layerIndex < layerSizes.Length; layerIndex++)
			{
				for (int neuronIndex = 0; neuronIndex < neurons[layerIndex].Length; neuronIndex++)
				{
					float value = 0f;

					for (int previousNeuronIndex = 0; previousNeuronIndex < neurons[layerIndex - 1].Length; previousNeuronIndex++)
					{
						value += weights[layerIndex][neuronIndex][previousNeuronIndex] * neurons[layerIndex - 1][previousNeuronIndex]; 
						//sum of all weights connections of this neuron weight their values in previous layer
					}

					neurons[layerIndex][neuronIndex] = (float)Math.Tanh(value); //Hyperbolic tangent activation
					//neurons[layerIndex][neuronIndex] = layerIndex == layers.Length - 1 ? value : ReLu(value); // ReLu activation
				}
			}

			// return output layer
			return neurons[layerSizes.Length - 1]; //return output layer
		}
		
		public override void Mutate()
		{
			for (int i = 1; i < weights.Length; i++)
			{
				for (int j = 0; j < weights[i].Length; j++)
				{
					for (int k = 0; k < weights[i][j].Length; k++)
					{
						if(Random.value < mutationChance)
						{
							float weight = weights[i][j][k];
							weights[i][j][k] = Random.Range(weight - maxFlatMutation, weight + maxFlatMutation);
						}
					}
				}
			}
		}

		public override string ToString()
		{
			var stringBuilder = new StringBuilder();

			for (int layerIndex = 0; layerIndex < layerSizes.Length; layerIndex++)
			{
				stringBuilder.Append("LAYER: index: ");
				stringBuilder.Append(layerIndex);
				stringBuilder.Append(" value: ");
				stringBuilder.Append(layerSizes[layerIndex]);
				stringBuilder.Append('\n');

				for (int neuronIndex = 0; neuronIndex < layerSizes[layerIndex]; neuronIndex++)
				{
					stringBuilder.Append("	NEURON: ");
					stringBuilder.Append(neuronIndex);
					stringBuilder.Append(" value: ");
					stringBuilder.Append(neurons[layerIndex][neuronIndex]);
					stringBuilder.Append('\n');

					if (layerIndex == 0) continue;
					for (int weightIndex = 0; weightIndex < layerSizes[layerIndex - 1]; weightIndex++)
					{
						stringBuilder.Append("		WEIGHT: ");
						stringBuilder.Append(weightIndex);
						stringBuilder.Append(" value: ");
						stringBuilder.Append(weights[layerIndex][neuronIndex][weightIndex]);
						stringBuilder.Append('\n');
					}
				}
			}

			return stringBuilder.ToString();
		}
	}
}