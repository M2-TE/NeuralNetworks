using NeuralNetworks;
using UnityEngine;

public class NeuralNetworkManager : MonoBehaviour
{
	[SerializeField] private NeuralNetworkInitSettings initSettings;
	private NeuralNetwork neuralNetwork;

	private void Start()
	{
		neuralNetwork = new NeuralNetworkNodebased(initSettings);
		Debug.Log(neuralNetwork);
	}
}
