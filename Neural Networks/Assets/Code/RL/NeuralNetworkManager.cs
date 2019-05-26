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
	[SerializeField] private GameObject agentPrefab;
	[SerializeField] private float xBounds;
	[SerializeField] private float yBounds;
	[SerializeField] private int agentCount;
#pragma warning restore 0649

	private readonly List<Agent> agents = new List<Agent>();
	private NeuralNetworkCompact newestNetwork;

	private void Start()
	{
		newestNetwork = new NeuralNetworkCompact
			(initSettings.layers,
			initSettings.weightMutationChance,
			initSettings.weightMaxMutationRange);

		for (int i = 0; i < 10; i++)
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
			agents.Add(agent);
		}
	}

	private void Update()
	{
		if (Input.GetKeyDown(KeyCode.Space))
		{
			NeuralNetwork fittestNetwork = agents[0].NN;
			for(int i = 1; i < agents.Count; i++)
			{
				if(agents[i].NN.Fitness > fittestNetwork.Fitness)
				{
					fittestNetwork = agents[i].NN;
				}
			}

			NeuralNetworkCompact cachedNN;
			for (int i = 1; i < agents.Count; i++)
			{
				cachedNN = new NeuralNetworkCompact(fittestNetwork as NeuralNetworkCompact);
				cachedNN.Mutate();
				agents[i].NN = cachedNN;
			}
		}
	}
}
