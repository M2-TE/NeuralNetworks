using System.Collections.Generic;
using System.Text;
using UnityEngine;
using Random = UnityEngine.Random;

public class QLearningPrototyping : MonoBehaviour
{
	private int[][] transitionMatrix;
	private double[][] rewardMatrix;
	private double[][] qualityMatrix;
	private int stateNum;
	private int goal;
	private double gamma;
	private double learnRate;
	private int maxEpochs;
	
	private void Awake()
	{
		Debug.Log("Begin Q-learning maze demo");
		Debug.Log("Setting up maze and rewards");
		stateNum = 12;
		transitionMatrix = CreateMaze();
		rewardMatrix = CreateReward();
		qualityMatrix = CreateQuality();

		Debug.Log("Analyzing maze using Q-learning");
		goal = 11;
		gamma = 0.5;
		learnRate = 0.5;
		maxEpochs = 1000;
		Train();

		Debug.Log("Done. Q matrix: ");
		Print(qualityMatrix);

		Debug.Log("Using Q to walk from cell 8 to 11");
		Walk(8, 11);
		Debug.Log("End demo");
	}

	private int[][] CreateMaze()
	{
		int[][] FT = new int[stateNum][];
		for (int i = 0; i < stateNum; ++i) FT[i] = new int[stateNum];
		FT[0][1] = FT[0][4] = FT[1][0] = FT[1][5] = FT[2][3] = 1;
		FT[2][6] = FT[3][2] = FT[3][7] = FT[4][0] = FT[4][8] = 1;
		FT[5][1] = FT[5][6] = FT[5][9] = FT[6][2] = FT[6][5] = 1;
		FT[6][7] = FT[7][3] = FT[7][6] = FT[7][11] = FT[8][4] = 1;
		FT[8][9] = FT[9][5] = FT[9][8] = FT[9][10] = FT[10][9] = 1;
		FT[11][11] = 1;  // Goal
		return FT;
	}

	private double[][] CreateReward()
	{
		double[][] R = new double[stateNum][];
		for (int i = 0; i < stateNum; ++i) R[i] = new double[stateNum];
		R[0][1] = R[0][4] = R[1][0] = R[1][5] = R[2][3] = -0.1;
		R[2][6] = R[3][2] = R[3][7] = R[4][0] = R[4][8] = -0.1;
		R[5][1] = R[5][6] = R[5][9] = R[6][2] = R[6][5] = -0.1;
		R[6][7] = R[7][3] = R[7][6] = R[7][11] = R[8][4] = -0.1;
		R[8][9] = R[9][5] = R[9][8] = R[9][10] = R[10][9] = -0.1;
		R[7][11] = 10.0;  // Goal
		return R;
	}

	private double[][] CreateQuality()
	{
		double[][] Q = new double[stateNum][];
		for (int i = 0; i < stateNum; ++i)
			Q[i] = new double[stateNum];
		return Q;
	}

	private List<int> GetPossNextStates(int s)
	{
		List<int> result = new List<int>();
		for (int j = 0; j < transitionMatrix.Length; ++j)
			if (transitionMatrix[s][j] == 1) result.Add(j);
		return result;
	}

	private int GetRandNextState(int s)
	{
		List<int> possNextStates = GetPossNextStates(s);
		int ct = possNextStates.Count;
		int idx = Random.Range(0, ct);
		return possNextStates[idx];
	}

	private void Train()
	{
		for (int epoch = 0; epoch < maxEpochs; epoch++)
		{
			int currState = Random.Range(0, rewardMatrix.Length);
			while (true)
			{
				int nextState = GetRandNextState(currState);
				List<int> possNextNextStates = GetPossNextStates(nextState);
				double maxQ = double.MinValue;
				for (int j = 0; j < possNextNextStates.Count; j++)
				{
					int nns = possNextNextStates[j];  // short alias
					double q = qualityMatrix[nextState][nns];
					if (q > maxQ) maxQ = q;
				}
				qualityMatrix[currState][nextState] = 
					((1 - learnRate) * qualityMatrix[currState][nextState]) 
					+ (learnRate * (rewardMatrix[currState][nextState] + (gamma * maxQ)));
				currState = nextState;
				if (currState == goal) break;
			}
		}
	}

	private void Walk(int start, int goal)
	{
		int curr = start; int next;
		Debug.Log(curr + "->");
		while (curr != goal)
		{
			next = ArgMax(qualityMatrix[curr]);
			Debug.Log(next + "->");
			curr = next;
		}
		Debug.Log("done");
	}

	private int ArgMax(double[] vector)
	{
		double maxVal = vector[0]; int idx = 0;
		for (int i = 0; i < vector.Length; ++i)
		{
			if (vector[i] > maxVal)
			{
				maxVal = vector[i]; idx = i;
			}
		}
		return idx;
	}

	private void Print(double[][] Q)
	{
		int ns = Q.Length;
		Debug.Log("[0] [1] . . [11]");
		var builder = new StringBuilder();
		for (int i = 0; i < ns; ++i)
		{
			for (int j = 0; j < ns; ++j)
			{
				builder.Append(Q[i][j].ToString("F2") + " ");
			}
			builder.AppendLine();
		}
		Debug.Log(builder.ToString());
	}
}

//https://msdn.microsoft.com/en-us/magazine/mt829710.aspx

//loop maxEpochs times
//  set currState = a random state
//  while currState != goalState
//	pick a random next-state but don't move yet
//    find largest Q for all next-next-states
//	update Q[currState][nextState] using Bellman
//    move to nextState
//  end-while
//end-loop