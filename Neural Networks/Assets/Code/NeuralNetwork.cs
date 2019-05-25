using System;
using System.Linq;
using UnityEngine;

namespace NeuralNetworks
{
	public abstract class NeuralNetwork : IComparable<NeuralNetwork>
	{
		public float Fitness;

		protected float mutationChance;
		protected float maxFlatMutation;

		public int CompareTo(NeuralNetwork other)
		{
			if (other == null) return 1;

			if (Fitness > other.Fitness) return 1;
			else if (Fitness < other.Fitness) return -1;
			else return 0;
		}

		public abstract void Mutate();
		public abstract float[] RequestDecision(float[] input);
		
		#region math utilities
		protected float ReLu(float x)
		{
			if (x < 0) return 0f;
			else return x;
		}

		protected void Softmax(ref float[] arr)
		{
			int i = 0;
			float sum = 0f;
			for (; i < arr.Length; i++)
			{
				//arr[i] = Mathf.Exp(arr[i]);
				sum += arr[i];
			}

			for (i = 0; i < arr.Length; i++)
			{
				arr[i] /= sum;
			}
		}

		protected void SoftmaxLinq(ref float[] z)
		{
			//float sum = arr.Sum();
			//arr = arr.Select(i => i / sum).ToArray();

			var z_exp = z.Select(Mathf.Exp);
			// [2.72, 7.39, 20.09, 54.6, 2.72, 7.39, 20.09]

			var sum_z_exp = z_exp.Sum();
			// 114.98

			var softmax = z_exp.Select(i => i / sum_z_exp);
			z = softmax.ToArray();
		}
		#endregion
	}

	[Serializable]
	public struct NeuralNetworkInitSettings
	{
		public int[] layers;
		
		[Range(0f, 1f)] public float weightMutationChance;
		public float weightMaxMutationRange;
	}
}