using NeuralNetworks;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Random = UnityEngine.Random;

public class NeuralNetworkManager : MonoBehaviour
{
#pragma warning disable 0649
	[SerializeField] private NeuralNetworkInitSettings initSettings;

	[SerializeField] private Camera cam;
	[SerializeField] private Agent agentPrefab;
	[SerializeField] private float xBounds;
	[SerializeField] private float yBounds;
	[SerializeField] private int agentCount;
#pragma warning restore 0649

	private List<Agent> agents;
	private NeuralNetworkCompact newestNetwork;

	private void Start()
	{
		newestNetwork = new NeuralNetworkCompact
			(initSettings.layers,
			initSettings.weightMutationChance,
			initSettings.weightMaxMutationRange);

		for (int i = 0; i < 5; i++)
		{
			newestNetwork.Mutate();
		}

		SpawnAgents();
	}

	private void SpawnAgents()
	{
		for(int i = 0; i < agentCount; i++)
		{
			var agent = Instantiate(agentPrefab, new Vector3(Random.Range(-xBounds, xBounds), Random.Range(-yBounds, yBounds), 0f), Quaternion.identity).GetComponent<Agent>();
			agent.cam = cam;
			agent.NN = new NeuralNetworkCompact(newestNetwork);
			agent.NN.Mutate();
		}
	}
}
