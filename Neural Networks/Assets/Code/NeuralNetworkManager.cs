using NeuralNetworks;
using UnityEngine;

public class NeuralNetworkManager : MonoBehaviour
{
	private NeuralNetworkCompact neuralNetworkCompact;

	private void Start()
	{
		var layers = new int[]
		{
			2, 3, 4, 2
		};
		neuralNetworkCompact = new NeuralNetworkCompact(layers);
		neuralNetworkCompact.Print();
	}
}
